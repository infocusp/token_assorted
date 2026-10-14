"""`deterministic: true` must actually configure determinism — and must not lie when it can't.

Stage-2 training is not bit-reproducible by default (the embedding gradient accumulates with
atomicAdd), so `deterministic: true` is the switch you flip to check whether a code change altered
training. These tests cover the wiring; the end-to-end proof is a GPU run, recorded in the
`_apply_determinism` docstring: two identical 12-micro-step runs went from differing at micro-step 8
to bit-identical across every step.

CPU-only, no model, no network.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from src import stage2_finetune as s2

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_is_on(tmp_path):
    """A config without the key gets it, and resolve_cfg writes it in
    explicitly so stage2_config.resolved.yaml and the MLflow params record it."""
    cfg = yaml.safe_load((_REPO_ROOT / "config" / "stage2_llama1b.yaml").read_text())
    assert cfg["deterministic"] is True
    assert s2._DETERMINISTIC_DEFAULT is True
    assert s2._CUBLAS_WORKSPACE_CONFIG == ":4096:8"
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"seed": 42}))
    assert s2.resolve_cfg(s2.build_argparser().parse_args(["--config", str(cfg_path)]))["deterministic"] is True


def test_cli_flag_exists_and_defaults_to_none():
    """default=None — otherwise the CLI would override the YAML value on every run; only an explicit
    --deterministic / --no-deterministic may win."""
    action = next(a for a in s2.build_argparser()._actions
                  if "--deterministic" in a.option_strings)
    assert action.default is None


def test_cli_flag_reaches_the_config(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": False, "seed": 42}))
    args = s2.build_argparser().parse_args(["--config", str(cfg_path), "--deterministic"])
    assert s2.resolve_cfg(args)["deterministic"] is True


def test_no_deterministic_opts_out(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": True, "seed": 42}))
    args = s2.build_argparser().parse_args(["--config", str(cfg_path), "--no-deterministic"])
    assert s2.resolve_cfg(args)["deterministic"] is False


def test_yaml_flag_survives_when_cli_flag_absent(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump({"deterministic": True, "seed": 42}))
    args = s2.build_argparser().parse_args(["--config", str(cfg_path)])
    assert s2.resolve_cfg(args)["deterministic"] is True


def test_asking_for_determinism_without_setup_is_an_error():
    """_apply_determinism can only run from main() (CUBLAS_WORKSPACE_CONFIG must precede CUDA
    init). A direct run_train()/run_smoke() caller must be told, not silently run non-deterministic
    training while its resolved config claims otherwise."""
    assert s2._DETERMINISM_APPLIED is False, "a prior test leaked global determinism state"
    with pytest.raises(SystemExit, match="_apply_determinism"):
        s2._assert_determinism_applied({"deterministic": True})
    with pytest.raises(SystemExit, match="_apply_determinism"):  # no key = the default = on
        s2._assert_determinism_applied({})


def test_no_assert_when_determinism_opted_out():
    s2._assert_determinism_applied({"deterministic": False})


def test_apply_determinism_sets_env_seeds_and_kernels():
    """Runs in a subprocess: use_deterministic_algorithms is process-global and would leak into
    every later test (and make the whole suite slower)."""
    code = (
        "import src.stage2_finetune as s2, os, torch, random\n"
        "s2._apply_determinism({'seed': 1234})\n"
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'\n"
        "assert torch.are_deterministic_algorithms_enabled()\n"
        "assert torch.backends.cudnn.deterministic is True\n"
        "assert torch.backends.cudnn.benchmark is False\n"
        "assert s2._DETERMINISM_APPLIED is True\n"
        "s2._assert_determinism_applied({'deterministic': True})\n"   # must NOT raise now
        "a = torch.randn(3); random_a = random.random()\n"
        "s2._apply_determinism({'seed': 1234})\n"
        "assert torch.equal(a, torch.randn(3)), 'torch seed not reapplied'\n"
        "assert random_a == random.random(), 'python random seed not reapplied'\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT,
                       capture_output=True, text=True)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr[-2000:]}"
    assert "OK" in r.stdout


def test_preset_cublas_workspace_is_respected():
    """`:16:8` is the documented low-memory alternative — don't stomp a deliberate choice."""
    code = (
        "import src.stage2_finetune as s2, os\n"
        "s2._apply_determinism({'seed': 1})\n"
        "assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == ':16:8', os.environ['CUBLAS_WORKSPACE_CONFIG']\n"
        "print('OK')\n"
    )
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":16:8")
    r = subprocess.run([sys.executable, "-c", code], cwd=_REPO_ROOT,
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, f"stdout={r.stdout}\nstderr={r.stderr[-2000:]}"
    assert "OK" in r.stdout


def test_determinism_runs_before_torch_is_imported_in_main():
    """The ordering is load-bearing: cuBLAS reads CUBLAS_WORKSPACE_CONFIG when it builds its
    handle, so _apply_determinism must come before _shim_accelerate() (which imports accelerate,
    hence torch). Guard the order in main()'s source so a future tidy-up can't silently swap it."""
    src = (_REPO_ROOT / "src" / "stage2_finetune.py").read_text()
    body = src[src.index("def main()"):]
    assert body.index("_apply_determinism") < body.index("_shim_accelerate"), (
        "main() must call _apply_determinism before _shim_accelerate imports torch"
    )
    # and the module itself must not import torch at module scope, or the env var is already late
    head = src[:src.index("def _shim_accelerate")]
    assert "\nimport torch" not in head and "\nfrom torch" not in head
