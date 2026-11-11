"""Bitwise differential tests for optimized initial Gaussian-ray generation."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

import numpy as np

from benchmark_performance import load_application


RAY_FIELDS = (
    "position",
    "direction",
    "power",
    "wavelength_m",
    "refractive_index",
    "polarization",
    "intensity",
    "time_s",
    "ray_id",
    "parent_id",
    "depth",
    "bundle_reflected",
)


def legacy_emit_initial_rays(source, target_count: int):
    """Frozen reference implementation from before the source optimization."""
    rays = source.emit()
    while rays.n_rays < target_count:
        source.radial_positions += 1
        rays = source.emit()
    if rays.n_rays == target_count:
        return rays
    keep_idx = np.floor(
        np.linspace(0, rays.n_rays, target_count, endpoint=False)
    ).astype(np.int64)
    return rays.subset(rays.xp.asarray(keep_idx, dtype=np.int64))


def to_host(value):
    module = type(value).__module__.split(".")[0]
    if module == "cupy":
        import cupy as cp

        return cp.asnumpy(value)
    return np.asarray(value)


def assert_bundles_bitwise_equal(test_case, expected, actual, label: str) -> None:
    test_case.assertEqual(expected.n_rays, actual.n_rays, label)
    for field in RAY_FIELDS:
        np.testing.assert_array_equal(
            to_host(getattr(expected, field)),
            to_host(getattr(actual, field)),
            err_msg=f"{label}.{field}",
        )


class SourceGenerationTests(unittest.TestCase):
    def test_current_20000_matches_saved_baseline_hashes(self) -> None:
        app, _ = load_application()
        baseline = json.loads(
            (Path(__file__).resolve().parent / "source_generation_baseline.json").read_text(
                encoding="utf-8"
            )
        )
        source = app.build_initial_source("numpy")
        rays = app.emit_initial_rays(source, 20_000)
        self.assertEqual(source.radial_positions, baseline["selected_radial_positions"])
        for field in RAY_FIELDS:
            array = np.ascontiguousarray(to_host(getattr(rays, field)))
            digest = hashlib.sha256()
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
            self.assertEqual(digest.hexdigest(), baseline["fields"][field]["sha256"], field)

    def test_ring_count_formula(self) -> None:
        app, _ = load_application()
        for radial_positions in (1, 2, 10, 55, 56, 81, 82, 100):
            source = app.build_initial_source("numpy")
            source.radial_positions = radial_positions
            rays = source.emit()
            self.assertEqual(
                rays.n_rays,
                1 + 3 * radial_positions * (radial_positions + 1),
            )

    def test_numpy_targets_and_ring_boundaries_are_bitwise_identical(self) -> None:
        app, _ = load_application()
        count_55 = 1 + 3 * 55 * 56
        count_56 = 1 + 3 * 56 * 57
        targets = (1, count_55 - 1, count_55, count_55 + 1, count_56, count_56 + 1, 20_000)
        for target_count in targets:
            with self.subTest(target_count=target_count):
                reference_source = app.build_initial_source("numpy")
                optimized_source = copy.deepcopy(reference_source)
                expected = legacy_emit_initial_rays(reference_source, target_count)
                actual = app.emit_initial_rays(optimized_source, target_count)
                self.assertEqual(reference_source.radial_positions, optimized_source.radial_positions)
                assert_bundles_bitwise_equal(self, expected, actual, f"numpy_{target_count}")

    @unittest.skipUnless(importlib.util.find_spec("cupy") is not None, "CuPy is not installed")
    def test_cupy_20000_is_bitwise_identical(self) -> None:
        app, _ = load_application()
        reference_source = app.build_initial_source("cupy")
        optimized_source = copy.deepcopy(reference_source)
        expected = legacy_emit_initial_rays(reference_source, 20_000)
        actual = app.emit_initial_rays(optimized_source, 20_000)
        self.assertEqual(reference_source.radial_positions, optimized_source.radial_positions)
        assert_bundles_bitwise_equal(self, expected, actual, "cupy_20000")


if __name__ == "__main__":
    unittest.main(verbosity=2)
