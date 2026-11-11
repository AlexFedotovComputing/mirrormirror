"""Run repeated cold Python subprocess benchmarks for project entry-point modes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from benchmark_performance import ROOT


APP = ROOT / "из+к-_оптимизация.py"


def command_for(mode: str) -> list[str]:
    outdir = ROOT / f"benchmark_e2e_output_{mode}"
    command = [sys.executable, str(APP), "--backend", "numpy", "--max-interactions", "20", "--outdir", str(outdir)]
    if mode == "compute":
        command.extend(("--no-plot", "--skip-csv"))
    elif mode == "interactive":
        command.extend(("--no-open-plot", "--skip-csv"))
    elif mode == "full":
        command.append("--no-open-plot")
    else:
        raise ValueError(mode)
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--modes", nargs="+", choices=("compute", "interactive", "full"), default=("compute", "interactive", "full"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = {"label": args.label, "repeats": args.repeats, "modes": {}}
    for mode in args.modes:
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            completed = subprocess.run(
                command_for(mode),
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=3600,
            )
            elapsed = time.perf_counter() - started
            if completed.returncode:
                raise RuntimeError(completed.stderr or completed.stdout)
            samples.append(elapsed)
        report["modes"][mode] = {
            "seconds": samples,
            "median_seconds": statistics.median(samples),
            "min_seconds": min(samples),
            "max_seconds": max(samples),
        }

    output = args.output or ROOT / f"benchmark_e2e_{args.label}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
