"""Capture stable hashes of the pre-optimization 20k initial ray bundle."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from benchmark_performance import ROOT, load_application
from test_source_generation import RAY_FIELDS, legacy_emit_initial_rays, to_host


def array_fingerprint(value):
    array = np.ascontiguousarray(to_host(value))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.tobytes())
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "sha256": digest.hexdigest(),
    }


def main() -> None:
    app, _ = load_application()
    source = app.build_initial_source("numpy")
    rays = legacy_emit_initial_rays(source, 20_000)
    report = {
        "implementation": "legacy_emit_initial_rays",
        "target_count": 20_000,
        "selected_radial_positions": source.radial_positions,
        "generated_count_before_trim": 1 + 3 * source.radial_positions * (source.radial_positions + 1),
        "n_rays": rays.n_rays,
        "total_power": float(np.sum(to_host(rays.power))),
        "fields": {field: array_fingerprint(getattr(rays, field)) for field in RAY_FIELDS},
    }
    path = ROOT / "source_generation_baseline.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
