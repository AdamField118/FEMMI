"""
tests/test_config.py
The layered YAML config (femmi/config.py): defaults load, a user file deep-merges
on top (only listed keys change), and dot-notation access works.

Run:
    python -m pytest tests/test_config.py -v
"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("yaml")
from femmi.config import load_config, DEFAULTS


def test_defaults_load():
    cfg = load_config(None)
    assert cfg.get("forward.nx") == DEFAULTS["forward"]["nx"]
    assert cfg.get("inverse.method") == "map"
    assert cfg.get("prior.kind") == "wiener"
    assert cfg.get("does.not.exist", "fallback") == "fallback"


def test_user_config_deep_merges(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("inverse:\n  method: sample\nprior:\n  kind: neural\n  neural:\n    n_pix: 64\n")
    cfg = load_config(str(p))
    # overridden keys
    assert cfg.get("inverse.method") == "sample"
    assert cfg.get("prior.neural.n_pix") == 64
    # untouched keys keep their defaults (deep merge, not replace)
    assert cfg.get("prior.neural.base") == DEFAULTS["prior"]["neural"]["base"]
    assert cfg.get("sampler.n_chains") == DEFAULTS["sampler"]["n_chains"]


def test_set_and_cli_coerce():
    from femmi.cli import _apply_overrides, _coerce
    cfg = load_config(None)
    _apply_overrides(cfg, ["inverse.method=sample", "forward.nx=8", "inverse.lam=0.3",
                           "data.flip_g2=true", "forward.radius=null"])
    assert cfg.get("inverse.method") == "sample"
    assert cfg.get("forward.nx") == 8 and isinstance(cfg.get("forward.nx"), int)
    assert cfg.get("inverse.lam") == 0.3
    assert cfg.get("data.flip_g2") is True
    assert cfg.get("forward.radius") is None
    assert _coerce("hello") == "hello"


def test_save_writes_samples_when_present(tmp_path):
    """_save persists individual posterior draws (node-aligned) into the .npz so
    plot_npz can render Figure-2-style samples + appearance frequency."""
    import numpy as np
    from femmi.pipeline import _save
    cfg = load_config(None)
    cfg.set("output.dir", str(tmp_path)); cfg.set("output.name", "r")
    nodes = np.zeros((5, 2))
    result = dict(kappa=np.ones(5), nodes=nodes, std=np.ones(5),
                  truth=np.zeros(5), samples=np.arange(15.0).reshape(3, 5))
    _save(cfg, result)
    d = np.load(str(tmp_path / "r.npz"))
    assert "samples" in d.files and d["samples"].shape == (3, 5)
    # opting out drops them
    cfg.set("output.save_samples", False); cfg.set("output.name", "r2")
    _save(cfg, result)
    assert "samples" not in np.load(str(tmp_path / "r2.npz")).files


def test_shipped_configs_are_valid():
    here = os.path.dirname(__file__)
    for name in ("default.yaml", "paper_artifacts.yaml"):
        path = os.path.join(here, "..", "configs", name)
        if not os.path.exists(path):
            continue
        cfg = load_config(path)
        assert cfg.get("prior.kind") in ("wiener", "tv", "sparse", "maxent", "neural")
        assert cfg.get("inverse.method") in ("map", "sample")


if __name__ == "__main__":
    import subprocess
    sys.exit(subprocess.call([sys.executable, "-m", "pytest", __file__, "-v"]))
