"""Every shipped Stage-1 config must parse.

`Config.from_dict` rejects unknown keys, so a YAML that names a field which was renamed or removed
raises at load time — but nothing exercised the shipped config files, so that only surfaced when
someone launched a run. These tests are pure YAML parsing plus `__post_init__` asserts (no model,
no tokenizer, no network), so they cost milliseconds.
"""
from pathlib import Path

import pytest

from src.utils import Config, load_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE1_CONFIGS = sorted((_REPO_ROOT / "config").glob("vqvae_*.yaml"))


# The configs the README ships (the repository tree lists exactly these three).
_RELEASE_STAGE1 = ["vqvae_llama.yaml", "vqvae_llama_repeat.yaml", "vqvae_llama_vanilla.yaml"]


@pytest.mark.parametrize("name", _RELEASE_STAGE1)
def test_release_configs_enable_mlflow(name):
    """MLflow logging is on in the shipped configs (like Stage 2); `--no-mlflow` turns it off."""
    assert load_config(str(_REPO_ROOT / "config" / name)).mlflow is True


def test_stage1_configs_found():
    assert _STAGE1_CONFIGS, "no config/vqvae_*.yaml found — has the config dir moved?"


@pytest.mark.parametrize("path", _STAGE1_CONFIGS, ids=lambda p: p.name)
def test_stage1_config_parses(path):
    cfg = load_config(str(path))
    # transformer_impl must resolve to a known arm whether or not the YAML mentions it (older
    # configs predate the field and rely on the default).
    assert cfg.transformer_impl in ("searchformer", "vanilla")


def test_transformer_impl_default_is_searchformer():
    """Load-bearing for checkpoint compatibility: a stored config from before this field existed
    must rebuild the vendored architecture, or every existing runs/*/best.pt changes meaning."""
    assert Config().transformer_impl == "searchformer"


def test_transformer_impl_rejects_unknown():
    with pytest.raises(AssertionError):
        Config(transformer_impl="nope")
