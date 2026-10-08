"""Fresh-process NumPy/Numba profiles with numerical parity before speed ratios."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from compare_cpu import compare


def run(args):
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("choose an empty output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    from femmi.protocol import source_provenance

    manifest = dict(
        source=source_provenance(),
        sources=args.sources,
        methods=args.methods,
        processes=args.processes,
        repeats=args.repeats,
        note="First Numba process per mesh case compiles into an empty cache; subsequent processes load it.",
    )
    (args.output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    driver = Path(__file__).with_name("profile_cpu.py")
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMBA_NUM_THREADS="1",
    )
    comparisons = []
    for sources in args.sources:
        for kind in args.methods:
            with tempfile.TemporaryDirectory(prefix="femmi-numba-") as cache:
                env["NUMBA_CACHE_DIR"] = cache
                for backend in ("numba", "numpy"):
                    for i in range(args.processes):
                        out = args.output / f"{kind}-{sources}-{backend}-{i}.json"
                        cmd = [
                            sys.executable,
                            str(driver),
                            "--kind",
                            kind,
                            "--sources",
                            str(sources),
                            "--repeats",
                            str(args.repeats),
                            "--output",
                            str(out),
                        ]
                        if backend == "numba":
                            cmd.append("--warm-numba")
                        print(out, flush=True)
                        with out.with_suffix(".log").open("w") as log:
                            subprocess.run(
                                cmd,
                                env=dict(env, FEMMI_BEM_BACKEND=backend),
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                check=True,
                            )
                for i in range(args.processes):
                    comparisons.append(
                        compare(
                            args.output / f"{kind}-{sources}-numpy-{i}.json",
                            args.output / f"{kind}-{sources}-numba-{i}.json",
                        )
                    )
                (args.output / "comparisons.json").write_text(
                    json.dumps(comparisons, indent=2) + "\n"
                )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--sources", type=int, nargs="+", default=[200, 1000])
    p.add_argument(
        "--methods",
        choices=["p3", "argyris", "hct"],
        nargs="+",
        default=["p3", "argyris", "hct"],
    )
    p.add_argument("--processes", type=int, default=3)
    p.add_argument("--repeats", type=int, default=3)
    args = p.parse_args()
    if min(args.sources) < 3 or args.processes < 2 or args.repeats < 2:
        p.error(
            "sources>=3, processes>=2 and repeats>=2 required for cold/cached/reused solves"
        )
    run(args)
