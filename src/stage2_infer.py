r"""Stage-2 inference: ask the fine-tuned LLM a math question and read its latent+text answer.

Takes a plain question, greedily decodes a response from the Stage-2 model and prints the question
and the raw generation, latent tokens included. The model writes its own latent tokens
(``<boLatent> <Latent-code-N> <eoLatent>``) followed by ordinary reasoning text and a final
``The answer is: <value>``; the Stage-1 VQ-VAE is not involved. It works for every
Stage-2 variant (Latent, CoT, Sol-Only; any Llama-3.2 base), since they all share the extended
tokenizer and the same prompt format.

It follows the evaluation protocol of ``stage2_eval.py``: the model is loaded by the same
``load_finetuned`` (full fine-tune or PEFT adapter, fp32), the prompt is ``[BOS] + question`` with no
chat template, and decoding is greedy with ``--max-new-tokens`` 512 and EOS stop, so a question gets
the same generation here as in the benchmark runs.

``--model`` takes a Stage-2 output dir (``runs/stage2-llama1b``) or a Hub repo
id (``infocusp/stage2-LLM``), holding a full fine-tune or a PEFT adapter. The question can also be
piped on stdin instead of ``--question``. A Hub repo id whose weights are not fully cached is
downloaded first (only the files loading needs), with a message and a progress bar on stderr; an
interrupted download is resumed. A cached repo is used as is, without checking the Hub for updates.
``--out`` additionally saves the full result as JSON: the generation with the latent tokens removed,
the final answer, the latent codes, the generated token count and whether it was cut off.

Example:
    python -m src.stage2_infer --model hf_models/stage2 \
        --question "Jean has 30 lollipops. Jean eats 2 of the lollipops. With the remaining lollipops, Jean wants to package 2 lollipops in one bag. How many bags can Jean fill?"
"""
from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import torch

from .data import ANSWER_MARKER, split_cot_solution
from .stage2_eval import load_finetuned
from .utils import select_device

# token strings of the latent vocabulary (config/stage2_*.yaml: code_token_fmt)
LATENT_CODE_RE = re.compile(r"<Latent-code-(\d+)>")


# files load_finetuned needs from a Hub repo: config/tokenizer/adapter JSONs, the weights (single
# file or shards), stage2_config.resolved.yaml (an adapter's base model) and non-JSON tokenizer files
HUB_FILE_PATTERNS = ["*.json", "*.safetensors", "*.yaml", "tokenizer*"]


def _cached_snapshot(model: str) -> Optional[str]:
    """Local snapshot folder of Hub repo ``model`` if its weights are fully cached, else None.

    An interrupted download leaves only a ``.incomplete`` blob, so the weights file is not in the
    snapshot and the repo counts as not cached; for a sharded model every shard must be there.
    """
    from huggingface_hub import try_to_load_from_cache

    def cached(filename: str) -> Optional[str]:
        # try_to_load_from_cache returns a path if cached, else None or a "known absent" sentinel
        path = try_to_load_from_cache(model, filename)
        return path if isinstance(path, str) else None

    for weights in ("model.safetensors", "adapter_model.safetensors"):    # full model, PEFT adapter
        if path := cached(weights):
            return os.path.dirname(path)
    if index := cached("model.safetensors.index.json"):                    # sharded model
        with open(index) as f:
            shards = set(json.load(f)["weight_map"].values())
        if all(cached(s) for s in shards):
            return os.path.dirname(index)
    return None


def resolve_model(model: str) -> str:
    """Return a local directory for ``--model``, downloading a Hub repo id first if needed.

    Local paths are returned as given. A Hub repo id resolves to its local snapshot folder, so
    ``load_finetuned`` gets a real directory (its PEFT-adapter check looks for a file on disk) and
    loading never downloads silently. A repo whose weights are not fully cached is downloaded first,
    only the files loading needs, with a message and the hub's progress bar on stderr
    (``snapshot_download`` resumes an interrupted download). A cached repo is used without
    contacting the Hub, so later updates to the repo are not picked up.
    """
    if os.path.exists(model) or model.count("/") != 1:  # a local path, not a Hub "org/name" id
        return model
    from huggingface_hub import HfApi, constants, snapshot_download
    from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError

    try:
        if path := _cached_snapshot(model):
            return path
        if constants.HF_HUB_OFFLINE:
            raise FileNotFoundError(f"{model} is not fully in the local Hugging Face cache and "
                                    f"HF_HUB_OFFLINE is set: unset it to download the model.")
        # also confirms the repo exists before we announce a download (e.g. a mistyped local path)
        info = HfApi().model_info(model, files_metadata=True)
        size_gb = sum(s.size or 0 for s in info.siblings
                      if any(fnmatch.fnmatch(s.rfilename, p) for p in HUB_FILE_PATTERNS)) / 1e9
        print(f"Downloading {model} ({size_gb:.1f} GB) from the Hugging Face Hub, not in the local "
              f"cache...", file=sys.stderr)
        path = snapshot_download(model, allow_patterns=HUB_FILE_PATTERNS)
    # a gated repo raises GatedRepoError, a subclass of RepositoryNotFoundError, and a private one
    # without access is reported as not found
    except (RepositoryNotFoundError, HFValidationError) as e:
        raise FileNotFoundError(f"{model!r} is neither a local directory nor a Hugging Face Hub "
                                f"repo you can access (if it is private or gated, log in with "
                                f"`hf auth login`).") from e
    print("Download complete.", file=sys.stderr)
    return path


@torch.no_grad()
def answer_question(model, tok, question: str, device, max_new_tokens: int = 512) -> Dict[str, Any]:
    """Greedy-decode one question. Returns the raw/clean generation, final answer and latent codes."""
    # base model, no chat template: prompt = [BOS] + question (matches training assembly)
    prompt = [tok.bos_token_id] + tok(question, add_special_tokens=False).input_ids
    input_ids = torch.tensor([prompt], device=device)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    out = model.generate(
        input_ids, attention_mask=torch.ones_like(input_ids),
        do_sample=False, num_beams=1,                                      # greedy
        max_new_tokens=max_new_tokens,
        eos_token_id=tok.eos_token_id, pad_token_id=pad_id,
    )
    gen = out[0, len(prompt):]
    # generated length = up to and including EOS; no EOS means the budget ran out mid-response
    eos_pos = (gen == tok.eos_token_id).nonzero()
    truncated = eos_pos.numel() == 0
    gen_ids: List[int] = gen[: gen.shape[0] if truncated else int(eos_pos[0, 0]) + 1].tolist()

    codes = [int(m.group(1)) for t in tok.convert_ids_to_tokens(gen_ids)
             if (m := LATENT_CODE_RE.fullmatch(t))]
    text = tok.decode(gen_ids, skip_special_tokens=True)                    # latent codes drop out
    _, solution = split_cot_solution(text)                                  # "The answer is: <value>"
    return {
        "question": question,
        "generation_raw": tok.decode(gen_ids, skip_special_tokens=False),
        "generation_text": text,
        "answer": solution[len(ANSWER_MARKER):].strip() if solution else None,
        "latent_codes": codes,
        "generated_tokens": len(gen_ids),
        "truncated": truncated,
        "max_new_tokens": max_new_tokens,
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", "--adapter", dest="model", required=True,
                   help="Stage-2 output dir or Hub repo id, holding a full fine-tune or a PEFT "
                        "adapter.")
    p.add_argument("--base-model", dest="base_model", default=None,
                   help="Only for PEFT adapters lacking stage2_config.resolved.yaml.")
    p.add_argument("--question", default=None, help="The problem to solve. Read from stdin when omitted.")
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=512)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="Optional JSON path for the full result.")
    return p


def main() -> None:
    p = build_argparser()
    args = p.parse_args()
    question = args.question
    if question is None:
        if sys.stdin.isatty():
            p.error("--question is required (or pipe the question on stdin)")
        question = sys.stdin.read()
    question = question.strip()
    if not question:
        p.error("the question is empty")

    device = select_device(args.device)
    # before disable_progress_bar() below, which also hides the hub's download bar
    model_dir = resolve_model(args.model)
    # keep the terminal to the question and the generation: hide the weight-loading progress bar and
    # load_finetuned's own "[eval] loaded ..." line. Loading keeps library warnings on, so a
    # checkpoint with missing or unexpected weights is still reported (errors always raise).
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    with contextlib.redirect_stdout(io.StringIO()):
        model, tok = load_finetuned(model_dir, args.base_model, device)
    hf_logging.set_verbosity_error()  # generate() warns about the checkpoint's max_length every call
    res = answer_question(model, tok, question, device, args.max_new_tokens)

    print(f"\n\nQuestion:\n{question}\n")
    print(f"Generation:\n{res['generation_raw']}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"model": args.model, **res}, f, indent=2)
        print(f"\n[stage2] wrote {args.out}")


if __name__ == "__main__":
    main()
