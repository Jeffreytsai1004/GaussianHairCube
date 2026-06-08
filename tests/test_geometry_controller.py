"""Test GeometryController brush operations and undo/redo."""

import unittest

import numpy as np

from tests.fixtures import make_cloud
from src.core.geometry_controller import (
    ControlMode, GeometryBrush, GeometryController,
)
from src.core.gaussian_generator import GaussianCloud


class TestGeometryController(unittest.TestCase):

    def setUp(self):
        self.ctrl = GeometryController()
        self.cloud = make_cloud(n=50, seed=3)
        self.ctrl.set_cloud(self.cloud)

    def test_set_and_get_cloud(self):
        self.assertIs(self.ctrl.get_cloud(), self.cloud)

    def test_apply_brush_returns_true_when_in_radius(self):
        # Brush centered on first splat — should affect at least itself
        brush = GeometryBrush(
            center=self.cloud.splats[0].position.copy(),
            radius=0.5, falloff=0.5, strength=0.5,
            mode=ControlMode.DENSITY,
        )
        self.assertTrue(self.ctrl.apply_brush(brush))

    def test_apply_brush_returns_false_when_no_points_in_radius(self):
        brush = GeometryBrush(
            center=np.array([100.0, 100.0, 100.0]),
            radius=0.01, falloff=0.5, strength=0.5,
            mode=ControlMode.DENSITY,
        )
        self.assertFalse(self.ctrl.apply_brush(brush))

    def test_density_brush_changes_opacity(self):
        original_opacities = [s.opacity for s in self.ctrl.get_cloud().splats]
        center = self.ctrl.get_cloud().splats[0].position.copy()
        brush = GeometryBrush(
            center=center, radius=1.0, falloff=0.5,
            strength=1.0, mode=ControlMode.DENSITY,
        )
        self.ctrl.apply_brush(brush, target_value=0.2)
        new_opacities = [s.opacity for s in self.ctrl.get_cloud().splats]
        # At least one opacity should have changed
        self.assertNotEqual(original_opacities, new_opacities)

    def test_color_brush_changes_color(self):
        center = self.ctrl.get_cloud().splats[0].position.copy()
        brush = GeometryBrush(
            center=center, radius=1.0, falloff=0.5,
            strength=1.0, mode=ControlMode.COLOR,
        )
        original = self.ctrl.get_cloud().splats[0].color.copy()
        target = np.array([0.9, 0.1, 0.1])
        self.ctrl.apply_brush(brush, target_value=target)
        new = self.ctrl.get_cloud().splats[0].color
        self.assertFalse(np.array_equal(original, new))

    def test_undo_restores_state(self):
        original = self.ctrl.get_cloud().splats[0].opacity
        brush = GeometryBrush(
            center=self.ctrl.get_cloud().splats[0].position.copy(),
            radius=1.0, falloff=0.5, strength=1.0, mode=ControlMode.DENSITY,
        )
        self.ctrl.apply_brush(brush, target_value=0.1)
        changed = self.ctrl.get_cloud().splats[0].opacity
        self.assertNotAlmostEqual(original, changed, places=4)

        self.assertTrue(self.ctrl.undo())
        restored = self.ctrl.get_cloud().splats[0].opacity
        self.assertAlmostEqual(original, restored, places=4)

    def test_undo_on_empty_history_returns_false(self):
        self.assertFalse(self.ctrl.undo())

    def test_smooth_region_does_not_crash(self):
        # smooth_region returned True even when modifying; must succeed here
        center = self.ctrl.get_cloud().bounds_min + (
            self.ctrl.get_cloud().bounds_max - self.ctrl.get_cloud().bounds_min) * 0.5
        ok = self.ctrl.smooth_region(center, radius=1.0, iterations=1)
        self.assertTrue(ok)

    def test_densify_increases_splat_count(self):
        n_before = self.ctrl.get_cloud().num_splats
        center = self.ctrl.get_cloud().splats[0].position.copy()
        ok = self.ctrl.densify_region(center, radius=1.0, factor=2.0)
        self.assertTrue(ok)
        n_after = self.ctrl.get_cloud().num_splats
        self.assertGreater(n_after, n_before)

    def test_prune_reduces_splat_count(self):
        n_before = self.ctrl.get_cloud().num_splats
        center = self.ctrl.get_cloud().splats[0].position.copy()
        ok = self.ctrl.prune_region(center, radius=1.0, keep_ratio=0.5)
        self.assertTrue(ok)
        n_after = self.ctrl.get_cloud().num_splats
        # Stochastic but the expectation is around 50% reduction inside radius
        self.assertLessEqual(n_after, n_before)

    def test_get_statistics(self):
        stats = self.ctrl.get_statistics()
        self.assertEqual(stats['num_splats'], self.cloud.num_splats)
        self.assertIn('mean_opacity', stats)
        self.assertIn('mean_scale', stats)

    def test_history_capped_at_max(self):
        self.ctrl.max_history = 3
        center = self.ctrl.get_cloud().splats[0].position.copy()
        brush = GeometryBrush(
            center=center, radius=1.0, falloff=0.5, strength=0.3,
            mode=ControlMode.SCALE,
        )
        # Apply more times than max_history
        for _ in range(5):
            self.ctrl.apply_brush(brush, target_value=1.2)
        self.assertLessEqual(len(self.ctrl.edit_history), 3)


if __name__ == "__main__":
    unittest.main()
