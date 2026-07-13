# Configuration & CLI

FEMMI is driven by a single YAML config that describes the whole pipeline. The
`femmi run` command loads it, builds the forward operator, gets the data, builds
the prior, runs the inverse (MAP or posterior sampling), and saves the outputs.

```bash
femmi run --config configs/default.yaml
```

Only list the keys you want to change; anything omitted falls back to
`femmi.config.DEFAULTS` (a deep-merge, so partial sections are fine).

## Config sections

| section | what it controls |
|---|---|
| `forward` | geometry (`catalog` vs structured `square`), mesh resolution, boundary elements, FEM–BEM `coupling` |
| `data`    | `source` (`synthetic` / `catalog_fits` / `frontier`), catalog params, optional circular mask |
| `inverse` | `method` (`map` / `sample`), regularization `lam` (null → auto), Wiener length, Morozov noise source |
| `prior`   | `kind` (`wiener` / `tv` / `sparse` / `maxent` / `neural`) + per-kind params |
| `sampler` | posterior-sampling method and its knobs (RTO draws; annealed-HMC ladder) |
| `output`  | `dir`, `name`, whether to save the `.npz` / figure |

The shipped [`configs/default.yaml`](https://github.com/AdamField118/FEMMI/blob/main/configs/default.yaml)
is annotated key-by-key and is the best reference.

## CLI overrides

Override any single value without editing the file — handy on a cluster:

```bash
femmi run --config configs/default.yaml \
    --set inverse.method=sample \
    --set prior.kind=neural \
    --set forward.nx=32
```

Values are coerced (`8` → int, `0.3` → float, `true`/`false` → bool, `null` → None).

## On a cluster (SLURM)

`scripts/femmi.sbatch` submits a run; the first argument is the config and any
further arguments are `--set` overrides:

```bash
sbatch scripts/femmi.sbatch configs/paper_artifacts.yaml
sbatch scripts/femmi.sbatch configs/my_run.yaml inverse.method=sample prior.kind=neural
```

It resolves the repo from `$SLURM_SUBMIT_DIR`, so submit from the repo root. Run
outputs default to `runs/` (git-ignored); if that directory is not writable the
pipeline falls back to `femmi_outputs/` (or a temp dir) with a warning rather than
dying at the final save.

## Reusing a config in a script

Purpose-built scripts (e.g. the paper-artifact recreation) load the *same* config
and reuse `femmi.pipeline`, so a script and a `femmi run` share exactly one
description of a run:

```python
from femmi.config import load_config
from femmi.pipeline import build_forward_and_data, build_prior

cfg = load_config("configs/paper_artifacts.yaml")
d   = build_forward_and_data(cfg)
prior = build_prior(cfg, d["ops"])
```
