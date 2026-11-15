import os
import unittest
import numpy as np
from mirrorlab.configurations import load_configuration
from mirrorlab.engine.parallel import trace_cpu
from mirrorlab.service import TraceRequest, static_manual_document
from mirrorlab.state import configuration_from_dataset


def canonical_segments(result):
    rows = []
    for block in result.segments:
        for i, surface in enumerate(block['surface']):
            rows.append((str(surface), tuple(float(block[k][i]) for k in
                ('x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'power'))))
    return sorted(rows)


class ParallelTests(unittest.TestCase):
    def test_no_old_ray_limit(self):
        self.assertEqual(TraceRequest(ray_count=50001).cpu_workers, 6)
        self.assertEqual(load_configuration().emit(50001).n_rays, 50001)

    def test_parallel_preserves_paths_power_and_identifiers(self):
        manual = static_manual_document('east_to_west')['datasets']['east_to_west']
        for config in (load_configuration('east_to_west'), configuration_from_dataset(manual)):
            rays = config.emit(217)
            args = dict(max_interactions=48, max_time_s=config.integration_time_s,
                        collision_mode='expected', **config.tracer_options)
            serial = trace_cpu(config.scene, rays, cpu_workers=1, **args)
            parallel = trace_cpu(config.scene, rays, cpu_workers=6, **args)
            self.assertEqual(parallel.diagnostics['cpu_workers'], min(6, os.cpu_count() or 1))
            self.assertNotIn(os.getpid(), parallel.diagnostics['worker_pids'])
            a, b = canonical_segments(serial), canonical_segments(parallel)
            self.assertEqual([x[0] for x in a], [x[0] for x in b])
            np.testing.assert_allclose([x[1] for x in a], [x[1] for x in b], atol=1e-12)
            for key, power in serial.detector_power_summary().items():
                self.assertAlmostEqual(power, parallel.detector_power_summary()[key], places=8)
            self.assertEqual(serial.diagnostics['unfinished_count'], parallel.diagnostics['unfinished_count'])
            self.assertEqual(len(np.unique(parallel.final_rays.ray_id)), parallel.final_rays.n_rays)
            parents = {}
            for block in parallel.segments + parallel.detector_hits:
                for ray, parent in zip(block['ray_id'], block['parent_id']):
                    if int(ray) in parents:
                        self.assertEqual(parents[int(ray)], int(parent))
                    parents[int(ray)] = int(parent)
            roots = set(map(int, rays.ray_id))
            for ray in parents:
                seen = set()
                while ray not in roots:
                    self.assertNotIn(ray, seen)
                    seen.add(ray)
                    ray = parents[ray]


if __name__ == '__main__':
    unittest.main()
