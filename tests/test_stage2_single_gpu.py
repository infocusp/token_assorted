"""Stage 2 is single-GPU: the FSDP/distributed path is gone and must stay gone.

It used to exist but was never reachable (every shipped config had ``enable_fsdp: false`` and no
launcher ever invoked ``torchrun``) and was never verified — it would have crashed at checkpoint
save on every rank != 0. These tests are the guard against it creeping back in, and against the
subtler failure: a config that still *asks* for sharding being trained single-GPU in silence.

Pure source/YAML inspection plus one ``train_config`` build — no model, no tokenizer, no network.
"""
from pathlib import Path

import pytest
import yaml

from src import stage2_finetune as s2

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE2_CONFIGS = sorted((_REPO_ROOT / "config").glob("stage2_*.yaml"))

# Names that only make sense with a distributed launcher. `enable_fsdp` is excluded: it is pinned
# to False in build_train_config because llama-cookbook's train() and get_dataloader_kwargs branch
# on it, so the one literal assignment there is load-bearing, not a leftover.
_BANNED_SOURCE_TOKENS = (
    "torch.distributed",
    "FullyShardedDataParallel",
    "init_process_group",
    "destroy_process_group",
    "fsdp_auto_wrap_policy",
    "apply_fsdp_checkpointing",
    "LOCAL_RANK",
    "WORLD_SIZE",
)


def test_stage2_configs_found():
    assert _STAGE2_CONFIGS, "no config/stage2_*.yaml found — has the config dir moved?"


@pytest.mark.parametrize("path", _STAGE2_CONFIGS, ids=lambda p: p.name)
def test_no_distributed_keys_in_shipped_configs(path):
    """The removed keys are silently ignored by the loader, so a reintroduced one would read as
    configured-and-honoured while doing nothing. Keep them out of the shipped configs."""
    cfg = yaml.safe_load(path.read_text()) or {}
    present = [k for k in s2._REMOVED_DISTRIBUTED_KEYS if k in cfg]
    assert not present, f"{path.name} still carries removed distributed key(s): {present}"


def test_source_has_no_distributed_machinery():
    src = (_REPO_ROOT / "src" / "stage2_finetune.py").read_text()
    # Ignore the docstring/comment lines that explain the absence.
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    found = [tok for tok in _BANNED_SOURCE_TOKENS if tok in code]
    assert not found, f"stage2_finetune.py reintroduced distributed machinery: {found}"


def test_argparser_has_no_fsdp_flag():
    opts = {s for a in s2.build_argparser()._actions for s in a.option_strings}
    assert "--enable-fsdp" not in opts


@pytest.mark.parametrize("key", s2._REMOVED_DISTRIBUTED_KEYS)
def test_truthy_removed_key_is_rejected_loudly(key):
    with pytest.raises(SystemExit, match="single-GPU only"):
        s2._reject_removed_distributed_keys({key: True})


@pytest.mark.parametrize("key", s2._REMOVED_DISTRIBUTED_KEYS)
def test_falsy_removed_key_still_loads(key):
    """Old configs and every saved stage2_config.resolved.yaml carry `<key>: false` — they must
    keep loading, or past runs become unreloadable."""
    s2._reject_removed_distributed_keys({key: False})


def test_train_config_is_single_gpu_and_keeps_clipping():
    """`gradient_clipping` used to be force-disabled whenever FSDP was on. That carve-out is gone:
    the config value must now reach cookbook's train_config verbatim."""
    s2._shim_accelerate()  # must precede any llama_cookbook import
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "stage2_llama1b.yaml").read_text())
    assert cfg["gradient_clipping"], "fixture config must have clipping ON for this test to bite"

    tc = s2.build_train_config(cfg)
    assert tc.enable_fsdp is False
    assert tc.low_cpu_fsdp is False
    assert tc.gradient_clipping == cfg["gradient_clipping"]
    assert tc.gradient_clipping_threshold == cfg["gradient_clipping_threshold"]
