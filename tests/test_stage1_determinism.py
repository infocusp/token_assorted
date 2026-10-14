"""Stage-1 `deterministic: true` must actually configure determinism — and must not lie when it can't.

Stage-1 training is already seed-deterministic (`set_seed` + the train loader's own generator), but on
GPU its kernels are not: vq-pytorch's kmeans-init `scatter_add_`, the `repeat_interleave` / SDPA
backward and cuDNN conv backward reorder float atomics run to run. `deterministic: true`
(`src/utils.py::apply_determinism`) is the switch that pins them. These tests cover the wiring, plus
an end-to-end CPU check that two identical `python -m src.train` runs come out bit-identical — the
seed half of reproducibility. The kernel half needs a GPU pair and is NOT covered here.

CPU-only. Anything that touches process-global state (`use_deterministic_algorithms`, the env var)
runs in a subprocess, so nothing leaks into the rest of the suite.
"""
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from src import train as s1
from src import utils as u

_REPO_ROOT = Path(__file__).resolve().parents[1]
_RECIPE = _REPO_ROOT / "config" / "vqvae_llama.yaml"


def _child_env(**extra):
    """No inherited CUBLAS_WORKSPACE_CONFIG (a shell export would break the asserts), never touch the
    shared GPU, and a fixed thread count — CPU reduction order depends on it."""
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", **extra)
    return env


def _run_py(code: str, **env) -> None:
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT, capture_output=True,
                       text=True, env=_child_env(**env))
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr[-2000:]}"
    assert "OK" in r.stdout


def test_default_is_on():
    """On by default since 2026-10-06: any two runs of one config must be comparable bit for bit.
    warn_only stays off — it would leave SDPA backward nondeterministic."""
    assert u.Config().deterministic is True
    assert u.Config().deterministic_warn_only is False
    assert u.load_config(str(_RECIPE)).deterministic is True
    assert u.CUBLAS_WORKSPACE_CONFIG == ":4096:8"


def test_cli_flag_exists_and_defaults_to_none():
    """default=None — apply_overrides skips None, so the YAML value (or the dataclass default) wins
    unless --deterministic / --no-deterministic is passed explicitly."""
    action = next(a for a in s1.build_argparser()._actions
                  if "--deterministic" in a.option_strings)
    assert action.default is None
    assert "deterministic" in s1._OVERRIDE_FIELDS  # apply_overrides only sees listed fields


def test_cli_flag_reaches_the_config(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": False}))
    for extra in ([], ["--smoke"]):  # smoke overrides apply before the CLI, so the flag still wins
        args = s1.build_argparser().parse_args(["--config", str(cfg_path), "--deterministic", *extra])
        assert s1.resolve_config(args).deterministic is True


def test_no_deterministic_opts_out(tmp_path):
    """With the default on, the CLI must be able to turn it off — and a YAML without the key is on."""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": True}))
    args = s1.build_argparser().parse_args(["--config", str(cfg_path), "--no-deterministic"])
    assert s1.resolve_config(args).deterministic is False
    cfg_path.write_text(yaml.safe_dump({"seed": 0}))
    assert s1.resolve_config(s1.build_argparser().parse_args(["--config", str(cfg_path)])).deterministic is True


def test_yaml_flag_survives_when_cli_flag_absent(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": True}))
    args = s1.build_argparser().parse_args(["--config", str(cfg_path)])
    assert s1.resolve_config(args).deterministic is True


def test_mlflow_switch_lives_in_the_config_and_flags_override(tmp_path):
    """Like Stage 2: the `mlflow` key decides (code default off, shipped configs on), the CLI flags
    override it, and --smoke forces it off so a wiring check never writes into the tracking store."""
    cfg_path = tmp_path / "c.yaml"
    cases = [(None, [], False),                       # key absent: off, so every other config is unchanged
             (True, [], True),                        # what the shipped configs do
             (True, ["--no-mlflow"], False),
             (None, ["--mlflow"], True),
             (True, ["--smoke"], False),
             (True, ["--smoke", "--mlflow"], True)]   # explicit flag beats the smoke default
    for yaml_val, argv, expected in cases:
        cfg_path.write_text(yaml.safe_dump({"seed": 0} if yaml_val is None else {"seed": 0, "mlflow": yaml_val}))
        args = s1.build_argparser().parse_args(["--config", str(cfg_path), *argv])
        assert s1.resolve_config(args).mlflow is expected, (yaml_val, argv)


def test_apply_determinism_sets_env_and_kernels():
    _run_py(
        "import os, torch, src.utils as u\n"
        "assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ\n"
        "u.apply_determinism()\n"
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'\n"
        "assert torch.are_deterministic_algorithms_enabled()\n"
        "assert not torch.is_deterministic_algorithms_warn_only_enabled()\n"
        "assert torch.backends.cudnn.deterministic is True\n"
        "assert torch.backends.cudnn.benchmark is False\n"
        "assert torch.backends.mkldnn.deterministic is True\n"
        "u.apply_determinism(warn_only=True)\n"
        "assert torch.is_deterministic_algorithms_warn_only_enabled()\n"
        "print('OK')\n"
    )


def test_preset_cublas_workspace_is_respected():
    """`:16:8` is the documented low-memory alternative — don't stomp a deliberate choice."""
    _run_py(
        "import os, src.utils as u\n"
        "u.apply_determinism()\n"
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8', os.environ['CUBLAS_WORKSPACE_CONFIG']\n"
        "print('OK')\n",
        CUBLAS_WORKSPACE_CONFIG=":16:8",
    )


def test_refuses_when_it_cannot_guarantee_determinism():
    """Two refusals, each raised BEFORE anything changes: (a) CUDA is already initialized, so cuBLAS
    may already have fixed its workspace config and a late setting would only look applied;
    (b) a preset CUBLAS_WORKSPACE_CONFIG that is not one of the two deterministic values (under
    warn_only it would otherwise just warn and train irreproducibly)."""
    _run_py(
        "import os, torch, src.utils as u\n"
        "real = torch.cuda.is_initialized\n"
        "torch.cuda.is_initialized = lambda: True\n"
        "try:\n"
        "    u.apply_determinism()\n"
        "    raise SystemExit('no raise after CUDA init')\n"
        "except RuntimeError as e:\n"
        "    assert 'before CUDA is initialized' in str(e), e\n"
        "assert 'CUBLAS_WORKSPACE_CONFIG' not in os.environ\n"
        "assert not torch.are_deterministic_algorithms_enabled()\n"
        "torch.cuda.is_initialized = real\n"
        "os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':0:0'\n"
        "try:\n"
        "    u.apply_determinism(warn_only=True)\n"
        "    raise SystemExit('no raise on a non-deterministic preset')\n"
        "except RuntimeError as e:\n"
        "    assert ':0:0' in str(e), e\n"
        "assert not torch.are_deterministic_algorithms_enabled()\n"
        "print('OK')\n"
    )


def test_determinism_applied_before_cuda_in_main():
    """The ordering is load-bearing: cuBLAS fixes CUBLAS_WORKSPACE_CONFIG at its first matmul, which
    assert_gpu_ok() runs. Guard the order in main()'s source so a tidy-up can't silently swap it."""
    src = (_REPO_ROOT / "src" / "train.py").read_text()
    # code only: main()'s comments name these calls too (no '#' occurs inside its string literals)
    body = "\n".join(line.split("#", 1)[0] for line in src[src.index("def main()"):].splitlines())
    first = body.index("apply_determinism(")
    for later in ("select_device(", "assert_gpu_ok(", "build_loaders(", "build_from_config("):
        assert first < body.index(later), f"main() must call apply_determinism before {later}"


def _assert_same(a, b, where: str) -> None:
    if torch.is_tensor(a):
        assert torch.is_tensor(b) and a.dtype == b.dtype and a.shape == b.shape, where
        assert torch.equal(a, b), f"{where} differs"
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), where
        for k in a:
            _assert_same(a[k], b[k], f"{where}.{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_same(x, y, f"{where}[{i}]")
    else:
        assert a == b, f"{where}: {a!r} != {b!r}"


def test_two_deterministic_runs_are_bit_identical(tmp_path, tokenizer):
    """End to end through `python -m src.train --deterministic`, on CPU: the release recipe at
    smoke size, run twice with one seed (A1, A2) plus a seed-1 control. A1 and A2 must agree bit for
    bit — every metrics.jsonl line, every model/optimizer tensor in last.pt (which includes the
    kmeans-initialized, EMA-updated codebook) — and the control must differ, or the comparison could
    never fail. threshold_ema_dead_code=2.0 forces dead-code restarts every step, so the random
    replacement-sampling path is exercised too. `tokenizer` (conftest) puts gpt2 in the HF cache, so
    the children can run offline. out_dir is tmp_path: the recipe's run_name would otherwise write
    into runs/vqvae-llama, the checkpoint behind the Stage-2 artifact.
    """
    cfg = u.apply_overrides(u.load_config(str(_RECIPE)), {
        **u.SMOKE_OVERRIDES,
        "steps": 8, "log_every": 1, "eval_every": 4, "ckpt_every": 8,
        "threshold_ema_dead_code": 2.0,
        "out_dir": str(tmp_path),
    })
    cfg_path = tmp_path / "det.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))

    procs = {
        name: subprocess.Popen(
            [sys.executable, "-m", "src.train", "--config", str(cfg_path), "--deterministic",
             "--run-name", name, "--seed", str(seed)],
            cwd=_REPO_ROOT, env=_child_env(HF_HUB_OFFLINE="1"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for name, seed in (("A1", 0), ("A2", 0), ("ctrl", 1))
    }
    for name, p in procs.items():
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, f"{name}: stdout={out[-2000:]}\nstderr={err[-2000:]}"
        assert "[determinism] ON" in out, name

    def rows(name):
        return (tmp_path / name / "metrics.jsonl").read_text().splitlines()

    a1, a2, ctrl = rows("A1"), rows("A2"), rows("ctrl")
    assert len(a1) == 10, a1  # 8 train rows (log_every=1) + val at steps 4 and 8
    assert sum('"split": "val"' in r for r in a1) == 2, a1
    # Raw lines, not parsed dicts: json writes floats repr-exact, and NaN != NaN would break `==`.
    assert a1 == a2
    assert a1 != ctrl, "the seed-1 control matched A1 — this comparison cannot fail, so it proves nothing"

    ck1, ck2 = (torch.load(tmp_path / n / "last.pt", map_location="cpu") for n in ("A1", "A2"))
    for key in ("model", "optimizer", "scheduler", "step", "best_metric"):  # not `config`: run_name differs
        _assert_same(ck1[key], ck2[key], key)
    for pt in tmp_path.rglob("*.pt"):
        pt.unlink()  # ~42 MB each and nothing else needs them
