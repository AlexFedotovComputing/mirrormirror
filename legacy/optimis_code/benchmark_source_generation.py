"""Repeatable timing and peak-memory benchmark for initial-ray generation."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import time

from benchmark_performance import ROOT, load_application, peak_working_set_bytes
from test_source_generation import legacy_emit_initial_rays


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--implementation", choices=("legacy", "current"), required=True)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    app, _ = load_application()
    function = legacy_emit_initial_rays if args.implementation == "legacy" else app.emit_initial_rays
    samples = []
    selected_radial_positions = []
    for _ in range(args.repeats):
        source = app.build_initial_source("numpy")
        started = time.perf_counter()
        rays = function(source, 20_000)
        samples.append(time.perf_counter() - started)
        selected_radial_positions.append(source.radial_positions)
        if rays.n_rays != 20_000:
            raise AssertionError(rays.n_rays)
        del rays, source
        gc.collect()

    report = {
        "implementation": args.implementation,
        "repeats": args.repeats,
        "seconds": samples,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "selected_radial_positions": selected_radial_positions,
        "peak_working_set_bytes": peak_working_set_bytes(),
    }
    output = args.output or ROOT / f"benchmark_source_{args.implementation}.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
