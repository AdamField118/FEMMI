"""Reproducible survey I/O smoke timings, not an accuracy/speed ranking.

Run from the repository root with PYTHONPATH=. and a fixed BLAS thread count.
All raw repetitions are saved; the first includes lazy imports/JIT cache loading.
"""

from datetime import datetime, timezone
import hashlib
import subprocess
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import tempfile
import time
import numpy as np
from astropy.table import Table
from femmi import map_mass, read_fits_catalog
from femmi.catalog import analytic_gaussian_shear
from femmi.survey import _software_versions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["p3", "argyris", "hct"])
    parser.add_argument("--shuffles", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.sources < 3 or args.repeats < 1 or args.shuffles < 2:
        parser.error("need >=3 sources, >=1 repetition, >=2 shuffles")
    rng = np.random.default_rng(118)
    xy = rng.uniform(-1.0, 1.0, (args.sources, 2))
    _, g1, g2 = analytic_gaussian_shear(xy, sigma=0.4, amp=0.1, center=(0.2, -0.1))
    dec = -30 + xy[:, 1] / 60
    ra = 35 + xy[:, 0] / (60 * np.cos(np.deg2rad(dec)))
    table = Table(
        dict(
            ra=ra,
            dec=dec,
            g1=g1,
            g2=-g2,
            weight=rng.uniform(0.5, 2, args.sources),
            ID=np.arange(args.sources),
        )
    )
    table["ra"].unit = table["dec"].unit = "deg"
    records = []
    with tempfile.TemporaryDirectory(prefix="femmi-survey-") as tmp:
        tmp = Path(tmp)
        path = tmp / "catalog.fits"
        table.write(path)
        for method in args.methods:
            for repeat in range(args.repeats):
                start = time.perf_counter()
                read_fits_catalog(path)
                read_seconds = time.perf_counter() - start
                wall_start = time.perf_counter()
                result = map_mass(
                    path,
                    method=method,
                    pixel_scale=0.1,
                    lam=0.3,
                    length=0.6,
                    mode=["E", "B"],
                    weight_col="weight",
                    id_col="ID",
                    centre=(35, -30),
                    create_counts_map=True,
                    create_snr=True,
                    num_shuffles=args.shuffles,
                    shuffle_type="orientation",
                    seed=118,
                    save_fits=True,
                    save_plots=False,
                    output_dir=tmp,
                    output_base_name=f"repetition{repeat}",
                )
                records.append(
                    dict(
                        method=method,
                        repetition=repeat,
                        wall_s=time.perf_counter() - wall_start,
                        read_only_s=read_seconds,
                        timings=result["timings"],
                        shape=result["maps"]["E"].shape,
                        diagnostics=result["metadata"]["diagnostics"],
                        saved_bytes=sum(
                            Path(f).stat().st_size for f in result["output_files"]
                        ),
                        peak_rss_process_kib=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss,
                    )
                )
    report = dict(
        schema_version=1,
        measured_at=datetime.now(timezone.utc).isoformat(),
        base_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                Path("femmi") / n
                for n in ("survey.py", "survey_output.py", "io.py", "mapping.py")
            ]
        },
        seed=118,
        sources=args.sources,
        shuffles=args.shuffles,
        repeats=args.repeats,
        software=_software_versions(),
        platform=platform.platform(),
        threads={
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
        backend=os.environ.get("FEMMI_BEM_BACKEND", "auto"),
        notes=[
            "Includes FITS E/B, null means/variance/SNR, counts, weights, coverage, source table, JSON.",
            "First repetition includes lazy imports/cache loading; timings do not isolate compilation.",
            "Assembly/factorization are combined; reconstruction and output evaluation are separate.",
            "Peak RSS is the cumulative process high-water mark in Linux KiB, not incremental per method.",
            "Three nulls exercise I/O only; this is not calibrated significance or a survey-scale benchmark.",
        ],
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
