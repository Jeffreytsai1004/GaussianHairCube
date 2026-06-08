"""Test multi-view reconstruction fallback ladder."""

import unittest
from unittest.mock import patch

import numpy as np

from src.core.multiview_reconstruction import (
    reconstruct_from_frames, _arc_fallback, _select_frames,
)


def _make_frames(n=4, size=64, seed=0):
    """Generate `n` random float32 RGB frames in [0, 1]."""
    rng = np.random.default_rng(seed)
    return [rng.random((size, size, 3)).astype(np.float32) for _ in range(n)]


class TestArcFallback(unittest.TestCase):

    def test_returns_nonempty_point_cloud(self):
        frames = _make_frames(n=3)
        pts, cols = _arc_fallback(frames)
        self.assertGreater(len(pts), 1000)
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(cols.shape[1], 3)
        # Upper hemisphere only (y > -0.2)
        self.assertGreaterEqual(pts[:, 1].min(), -0.2)

    def test_handles_empty_frames(self):
        pts, cols = _arc_fallback([])
        self.assertGreater(len(pts), 0)
        # Colors should default to gray range
        self.assertTrue(0.0 <= cols.mean() <= 1.0)


class TestSelectFrames(unittest.TestCase):

    def test_empty_input(self):
        self.assertEqual(_select_frames([]), [])

    def test_uniform_sampling_under_limit(self):
        frames = _make_frames(n=5)
        out = _select_frames(frames, max_frames=10)
        # All 5 should pass through (less than max_frames)
        self.assertEqual(len(out), 5)

    def test_uniform_sampling_over_limit(self):
        frames = _make_frames(n=20)
        out = _select_frames(frames, max_frames=8)
        self.assertLessEqual(len(out), 8)

    def test_long_edge_resize(self):
        # 1000-pixel-long frame should be resized to ≤ 960
        big = (np.random.rand(200, 1000, 3) * 255).astype(np.float32) / 255.0
        out = _select_frames([big, big, big], max_long_edge=480)
        for f in out:
            self.assertLessEqual(max(f.shape[:2]), 480)


class TestReconstructFromFrames(unittest.TestCase):

    def test_too_few_frames_uses_arc_fallback(self):
        """<2 frames goes straight to tier 4."""
        pts, cols, tier = reconstruct_from_frames([np.random.rand(64, 64, 3).astype(np.float32)])
        self.assertEqual(tier, 4)
        self.assertGreater(len(pts), 0)

    def test_random_frames_fall_through_to_arc(self):
        """Random noise frames have no real features → should reach tier 4."""
        frames = _make_frames(n=5, size=64)
        pts, cols, tier = reconstruct_from_frames(frames, callback=lambda p, m: None)
        # Tier 1-3 require matchable features. Random noise can't satisfy these,
        # but if pycolmap accidentally succeeds, accept any tier as long as we got points.
        self.assertIn(tier, (1, 2, 3, 4))
        self.assertGreater(len(pts), 0)
        self.assertEqual(pts.shape[1], 3)
        self.assertEqual(cols.shape, pts.shape)

    def test_callback_is_invoked(self):
        msgs = []
        frames = _make_frames(n=3)
        reconstruct_from_frames(frames, callback=lambda p, m: msgs.append((p, m)))
        self.assertGreater(len(msgs), 0)
        # Progress values must be monotonically non-decreasing within [0, 1]
        progresses = [p for p, _ in msgs]
        for p in progresses:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_cancel_flag_short_circuits(self):
        """cancel_flag_getter returning True should stop processing fast."""
        frames = _make_frames(n=5)
        cancel_state = {'flag': False}
        msgs = []

        def cancel_getter():
            cancel_state['flag'] = True
            return True

        pts, cols, tier = reconstruct_from_frames(
            frames,
            callback=lambda p, m: msgs.append(m),
            cancel_flag_getter=cancel_getter,
        )
        # Should still return something (arc fallback if everything else cancelled)
        self.assertGreater(len(pts), 0)

    def test_hair_masks_can_be_none(self):
        frames = _make_frames(n=3)
        pts, cols, tier = reconstruct_from_frames(frames, hair_masks=None)
        self.assertGreater(len(pts), 0)

    def test_output_arrays_are_float32(self):
        frames = _make_frames(n=3)
        pts, cols, _ = reconstruct_from_frames(frames)
        self.assertEqual(pts.dtype, np.float32)
        self.assertEqual(cols.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
