"""
femmi/cli.py
The `femmi` command line: one entry point that runs a whole pipeline from a config.

    femmi run --config configs/default.yaml
    femmi run --config my_run.yaml --set inverse.method=sample --set prior.kind=neural
    femmi train-prior --config my_run.yaml

`--set key=value` overrides any config value with dot-notation, so a run is fully
described by a config file plus optional overrides -- no bespoke flags per script.
Installed as the `femmi` console script (see pyproject [project.scripts]).
"""

from __future__ import annotations
import argparse
import sys

from .config import load_config


def _coerce(v: str):
    """Turn a CLI string into an int / float / bool / None / str."""
    low = v.lower()
    if low in ("none", "null"):
        return None
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def _apply_overrides(cfg, sets):
    for s in sets or []:
        key, sep, val = s.partition("=")
        if not sep:
            raise SystemExit(f"--set expects key=value, got {s!r}")
        cfg.set(key.strip(), _coerce(val.strip()))


def _load(args):
    cfg = load_config(args.config)
    _apply_overrides(cfg, args.set)
    return cfg


def cmd_run(args):
    from .pipeline import run
    run(_load(args))


def cmd_map(args):
    import numpy as np
    from .mapping import MapperConfig, map_mass
    from .io import FlatCatalog
    with np.load(args.catalogue, allow_pickle=False) as data:
        c = FlatCatalog(*(data[k] for k in ('x','y','g1','g2','weight')),
            units=str(data['units']) if 'units' in data else 'arcmin',
            z=data['z'] if 'z' in data else None)
    result=map_mass(c,MapperConfig.from_file(args.config))
    result.save(args.output)
    print(result.diagnostics)


def cmd_train_prior(args):
    cfg = _load(args)
    from .neural_prior.train import train_score_model
    n = cfg.get("prior.neural", {}) or {}
    train_score_model(n_pix=n.get("n_pix", 32), base=n.get("base", 16),
                      steps=cfg.get("prior.neural.steps", 8000),
                      hybrid=n.get("hybrid", False),
                      train_data=n.get("train_data", "synthetic"),
                      data_dir=n.get("data_dir"),
                      map_glob=n.get("map_glob"), pool_maps=n.get("pool_maps", 512),
                      verbose=True)


def build_parser():
    p = argparse.ArgumentParser(prog="femmi",
                                description="FEMMI: weak-lensing mass reconstruction, config-driven.")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", type=str, default=None,
                        help="YAML config (see configs/default.yaml); omit for built-in defaults")
        sp.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override any config value, e.g. --set inverse.method=sample")

    m = sub.add_parser("map", help="production shear catalogue MAP (P3, Argyris or HCT)")
    m.add_argument('--catalogue', required=True, help='NPZ: x,y,g1,g2,weight in arcmin')
    m.add_argument('--config', required=True, help='explicit MapperConfig YAML')
    m.add_argument('--output', required=True, help='output NPZ')
    m.set_defaults(func=cmd_map)

    r = sub.add_parser("run", help="build forward, get data, run MAP or sampling, save (the pipeline)")
    add_common(r); r.set_defaults(func=cmd_run)

    t = sub.add_parser("train-prior", help="train the neural score prior on synthetic maps")
    add_common(t); t.set_defaults(func=cmd_train_prior)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
