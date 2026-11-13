"""Reproducible, core-only timing and complete result snapshots for regression checks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import scipy
from mirrorlab.calibration import empirical, physical, reconstruction, validation
from mirrorlab.calibration.manual_refinement import refine_manual
from mirrorlab.configurations import load_configuration
from mirrorlab.manual_schema import import_manual
from mirrorlab.state import encode


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode", choices=["full", "manual", "both"], default="both")
    parser.add_argument("--direction", choices=["east_to_west", "west_to_east", "both"], default="both")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    print("Modules loaded; benchmark ready", flush=True)
    summary = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, runs=[])
    directions = ["east_to_west", "west_to_east"] if args.direction == "both" else [args.direction]
    for mode in (["full", "manual"] if args.mode == "both" else [args.mode]):
        for direction in directions:
            config = load_configuration(direction, "real")
            if mode == "manual":
                filename = "west_manual_corrected_survey.json" if direction == "east_to_west" else "east_recovered_centers.json"
                dataset = import_manual(ROOT / "data/manual_refinement" / filename)["datasets"][direction]
            physical._FIELD_CACHE.clear()
            if hasattr(physical, "_INCOMING_CACHE"):
                physical._INCOMING_CACHE.clear()
            started = perf_counter()
            previous = None
            def progress(event):
                nonlocal previous
                stage = event.get("stage")
                if stage != previous:
                    print(f"{mode} {direction}: {stage} ({perf_counter()-started:.3f}s)", flush=True)
                    previous = stage
            result = reconstruction.reconstruct(config, root=ROOT, progress=progress) if mode == "full" else refine_manual(dataset, progress=progress)
            elapsed = perf_counter() - started
            (args.output / f"{mode}-{direction}.json").write_text(
                json.dumps(encode(result), ensure_ascii=False, allow_nan=False), encoding="utf-8")
            summary["runs"].append(dict(mode=mode, direction=direction, seconds=elapsed, status=result["status"]))
            (args.output / "timing.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"DONE {mode} {direction}: {elapsed:.3f}s", flush=True)


if __name__ == "__main__":
    main()
