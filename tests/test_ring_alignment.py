"""Physical invariants of the corrected distributor, including saved fits."""
import copy
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from mirrorlab.calibration.physical import incoming_at_mirrors, validate_reflections
from mirrorlab.configurations import load_configuration
from mirrorlab.engine import RayTracer
from mirrorlab.ring_alignment import apply_ring_alignment
from mirrorlab.ring_migration import upgrade_ring_document
from mirrorlab.state import (configuration_from_dataset, encode, load_document,
                             save_document, snapshot, validate_document)

ROOT = Path(__file__).resolve().parents[1]


class RingAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch('mirrorlab.ring_alignment.apply_ring_alignment'):
            cls.before = load_configuration('west_to_east')
        cls.after = load_configuration('west_to_east')

    def test_downstream_hardware_and_all_incident_axes_are_preserved(self):
        old = {e.name: encode(e) for e in self.before.scene.elements}
        new = {e.name: encode(e) for e in self.after.scene.elements}
        for name in old:
            if name.startswith(('MC', 'BUNDLE_', 'Screen', 'Cylindrical')):
                self.assertEqual(old[name], new[name], name)
        before = incoming_at_mirrors(self.before)
        after = incoming_at_mirrors(self.after)
        self.assertEqual(len(after), 112)
        for key in before:
            self.assertEqual(after[key]['status'], 'available', key)
            np.testing.assert_allclose(before[key]['point_m'], after[key]['point_m'], atol=0, rtol=0)
            np.testing.assert_allclose(before[key]['direction'], after[key]['direction'], atol=1e-12, rtol=0)

    def test_ring_is_reflected_and_triangular_apertures_keep_their_vertices(self):
        old = {s.name: s for s in self.before.scene.build_surfaces()}
        new = {s.name: s for s in self.after.scene.build_surfaces()}
        reflection = np.diag([-1., 1., 1.])
        for name, surface in old.items():
            if not name.startswith(('ML', 'MR', 'PL', 'PR')):
                continue
            changed = new[name]
            np.testing.assert_allclose(changed.center, reflection @ surface.center, atol=1e-13, rtol=0)
            np.testing.assert_allclose(changed.normal, reflection @ surface.normal, atol=1e-13, rtol=0)
            if surface.shape == 'triangle':
                def vertices(s):
                    xy = np.asarray(s.triangle_vertices)
                    return s.center + xy[:, :1] * s._u + xy[:, 1:] * s._v
                np.testing.assert_allclose(vertices(changed), vertices(surface) @ reflection, atol=1e-13, rtol=0)

    def test_periscope_is_vertical_without_changing_length(self):
        old = {s.name: s for s in self.before.scene.build_surfaces()}
        new = {s.name: s for s in self.after.scene.build_surfaces()}
        delta = new['MP1'].center - new['MP2'].center
        self.assertLess(abs(delta[0]), 1e-12)
        self.assertLess(delta[1], 0)
        self.assertAlmostEqual(np.linalg.norm(delta), np.linalg.norm(old['MP1'].center-old['MP2'].center), places=12)
        self.assertAlmostEqual(self.before.source.waist_position[2], self.after.source.waist_position[2], places=12)

    def test_full_beam_and_perimeter_reach_all_four_rods(self):
        config = self.after
        for perimeter in (False, True):
            rays = config.emit(4000 if not perimeter else 512)
            if perimeter:
                theta = np.linspace(0, 2*np.pi, rays.n_rays, endpoint=False)
                radius = config.source.waist_radius * config.source.cutoff_ratio
                rays.position[:] = config.source.waist_position + radius*np.c_[np.cos(theta), np.sin(theta), np.zeros(len(theta))]
            result = RayTracer(config.scene, max_interactions=48, max_time_s=config.integration_time_s,
                               collision_mode='expected', **config.tracer_options).trace(rays)
            hits = Counter(name for block in result.segments for name in block['surface'])
            for rod in range(1, 5):
                self.assertEqual(hits[f'MC{rod}'], rays.n_rays)
            if not perimeter:
                self.assertEqual(len([name for name in hits if name.startswith('BUNDLE_')]), 112)
            self.assertEqual(result.diagnostics['unfinished_count'], 0)

    def test_correction_is_idempotent_and_does_not_touch_disabled_route(self):
        config = copy.deepcopy(self.after)
        before = snapshot(config)
        apply_ring_alignment(config)
        self.assertEqual(before, snapshot(config))
        config = load_configuration('east_to_west')
        before = snapshot(config)
        apply_ring_alignment(config)
        self.assertEqual(before, snapshot(config))

    def test_saved_fit_upgrade_is_reversible_lossless_and_portable(self):
        original = load_document(ROOT/'mirrorlab'/'saved restored'/'east1-mirrorlab-configuration.json')
        before_copy = copy.deepcopy(original)
        upgraded = upgrade_ring_document(original)
        self.assertEqual(original, before_copy)
        validate_document(upgraded)
        old = original['datasets']['west_to_east']
        new = upgraded['datasets']['west_to_east']
        for key in ('mirrors', 'display_frame', 'observations', 'assignments'):
            self.assertEqual(old[key], new[key], key)
        self.assertEqual(new['ring_alignment_previous_optics'], old['optics_snapshot'])
        self.assertEqual(upgraded['configuration_version'], original['configuration_version']+1)
        self.assertEqual(new['ring_alignment_validation']['validated_before'], new['ring_alignment_validation']['validated_after'])
        self.assertEqual(upgrade_ring_document(upgraded), upgraded)
        before_field = incoming_at_mirrors(configuration_from_dataset(old))
        after_field = incoming_at_mirrors(configuration_from_dataset(new))
        for key in before_field:
            self.assertEqual(before_field[key]['status'], after_field[key]['status'], key)
            np.testing.assert_allclose(before_field[key]['direction'], after_field[key]['direction'], atol=1e-12, rtol=0)
        with TemporaryDirectory() as directory:
            path = Path(directory)/'corrected.json'
            save_document(upgraded, path)
            self.assertEqual(load_document(path), upgraded)

    def test_manual_real_retains_all_physical_paths_without_larger_source(self):
        from mirrorlab.manual_schema import import_manual, BUNDLE
        from mirrorlab.ring_migration import upgrade_ring_dataset
        with patch('mirrorlab.ring_migration.upgrade_ring_dataset', side_effect=lambda d: d):
            original = import_manual(BUNDLE/'east_recovered_centers.json')['datasets']['west_to_east']
        upgraded = upgrade_ring_dataset(original)
        self.assertEqual(upgraded['mirrors'], original['mirrors'])
        self.assertEqual(upgraded['display_frame'], original['display_frame'])
        old, new = [configuration_from_dataset(d) for d in (original, upgraded)]
        for field in ('waist_radius', 'cutoff_ratio', 'peak_intensity'):
            self.assertEqual(getattr(old.source, field), getattr(new.source, field))
        before, after = [validate_reflections(c, original['mirrors']) for c in (old, new)]
        self.assertEqual(before['validated_count'], 112)
        self.assertEqual(after['validated_count'], 112)
        for key, first in before['mirrors'].items():
            second = after['mirrors'][key]
            np.testing.assert_allclose(first['hit_m'], second['hit_m'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(first['outgoing_direction'], second['outgoing_direction'], atol=1e-12, rtol=0)
            self.assertEqual(first['next_surfaces'], second['next_surfaces'])
            for a, b in zip(first['reflected_trajectories'], second['reflected_trajectories']):
                self.assertEqual(a['terminal_surface'], b['terminal_surface'])
                np.testing.assert_allclose(a['points_m'], b['points_m'], atol=1e-12, rtol=0)

    def test_unsupported_geometry_fails_without_partial_mutation(self):
        config = copy.deepcopy(self.before)
        config.scene.elements = [e for e in config.scene.elements if e.name != 'MC1']
        original = snapshot(config)
        with self.assertRaisesRegex(ValueError, 'missing upstream'):
            apply_ring_alignment(config)
        self.assertEqual(snapshot(config), original)


if __name__ == '__main__':
    unittest.main()
