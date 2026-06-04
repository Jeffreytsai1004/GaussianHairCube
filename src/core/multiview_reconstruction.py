"""
Multi-View Reconstruction Module
=================================

Replaces the fake camera-pose estimation in gaussian_generator.py with a real
4-tier fallback pipeline:

  Tier 1 — pycolmap full incremental SfM   (≥10 frames registered, ≥500 points)
  Tier 2 — pycolmap partial SfM            (5-9 frames registered, ≥200 points)
  Tier 3 — OpenCV relative pose chaining   (pycolmap unavailable / failed)
  Tier 4 — Arc rotation fallback           (synthetic hemisphere, always succeeds)

Public interface
----------------
    reconstruct_from_frames(frames, hair_masks, callback, cancel_flag_getter)
    → (points: np.ndarray (N,3), colors: np.ndarray (N,3), tier: int)
"""

import tempfile
import os
import shutil
import logging

import numpy as np
import cv2

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Frame selection helper
# ---------------------------------------------------------------------------

def _select_frames(frames, max_frames=40, max_long_edge=960):
    """
    Uniformly sample up to *max_frames* from the input list.

    Frames with a Laplacian variance below 20 (very blurry) are discarded.
    Each kept frame is resized so its long edge ≤ *max_long_edge*.

    Returns
    -------
    list of float32 HxWx3 numpy arrays
    """
    if not frames:
        return []

    # Uniform temporal sampling
    if len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        sampled = [frames[i] for i in indices]
    else:
        sampled = list(frames)

    def _resize(frame):
        h, w = frame.shape[:2]
        long_edge = max(h, w)
        if long_edge > max_long_edge:
            scale = max_long_edge / long_edge
            new_w, new_h = int(w * scale), int(h * scale)
            return cv2.resize(frame, (new_w, new_h)).astype(np.float32)
        return frame.astype(np.float32)

    selected = []
    for frame in sampled:
        # Blur check via Laplacian variance
        gray = cv2.cvtColor(
            (frame * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY
        )
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 20:
            continue  # skip truly blurry frames
        selected.append(_resize(frame))

    # If blur filter removed too many frames, fall back to all sampled frames
    if len(selected) < max(2, len(sampled) // 2):
        selected = [_resize(f) for f in sampled]

    return selected


# ---------------------------------------------------------------------------
# Tier 1 & 2 — pycolmap incremental SfM
# ---------------------------------------------------------------------------

def _run_pycolmap_sfm(frames, hair_masks, callback, cancel_getter):
    """
    Run pycolmap incremental SfM on the supplied frames.

    Returns
    -------
    (points_3d : np.ndarray Nx3,
     colors    : np.ndarray Nx3,
     num_registered : int)

    Raises RuntimeError on any failure so the tier fallback can engage.
    """
    try:
        import pycolmap
    except ImportError:
        raise RuntimeError("pycolmap not installed")

    import pathlib

    # 1. Select frames
    selected = _select_frames(frames, max_frames=40, max_long_edge=960)
    if len(selected) < 3:
        raise RuntimeError(f"Only {len(selected)} usable frames (need ≥3)")

    if callback:
        callback(0.10, f"选取了 {len(selected)} 帧用于三维重建")

    # 2. Write frames to a temp directory as JPEG
    tmpdir = tempfile.mkdtemp(prefix="ghc_sfm_")
    try:
        image_dir = pathlib.Path(tmpdir) / "images"
        image_dir.mkdir()
        db_path = pathlib.Path(tmpdir) / "database.db"
        output_path = pathlib.Path(tmpdir) / "sparse"
        output_path.mkdir()

        for i, frame in enumerate(selected):
            img_uint8 = (frame * 255).clip(0, 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            # Enforce resolution cap
            h, w = img_bgr.shape[:2]
            long_edge = max(h, w)
            if long_edge > 960:
                scale = 960 / long_edge
                img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
            cv2.imwrite(
                str(image_dir / f"frame_{i:04d}.jpg"),
                img_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

        if cancel_getter and cancel_getter():
            raise RuntimeError("Cancelled by user")
        if callback:
            callback(0.20, "正在提取特征点...")

        # 3. Feature extraction — try different API shapes across pycolmap versions
        try:
            # pycolmap >= 0.5 style
            pycolmap.extract_features(
                database_path=db_path,
                image_path=image_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                sift_options=pycolmap.SiftExtractionOptions(max_num_features=8192),
            )
        except (AttributeError, TypeError):
            try:
                # Older / alternative signatures
                pycolmap.extract_features(
                    database_path=str(db_path),
                    image_path=str(image_dir),
                )
            except Exception as exc:
                raise RuntimeError(f"pycolmap.extract_features failed: {exc}") from exc

        if cancel_getter and cancel_getter():
            raise RuntimeError("Cancelled by user")
        if callback:
            callback(0.35, "正在匹配特征点...")

        # 4. Sequential matching (suitable for video where frames are temporally ordered)
        try:
            pycolmap.match_sequential(
                database_path=db_path,
                matching_options=pycolmap.SequentialMatchingOptions(
                    overlap=10, loop_detection=False
                ),
            )
        except (AttributeError, TypeError):
            try:
                pycolmap.match_sequential(database_path=str(db_path))
            except Exception as exc:
                raise RuntimeError(f"pycolmap.match_sequential failed: {exc}") from exc

        if cancel_getter and cancel_getter():
            raise RuntimeError("Cancelled by user")
        if callback:
            callback(0.50, "正在运行增量式三维重建 (SfM)...")

        # 5. Incremental mapping
        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=image_dir,
                output_path=output_path,
                options=pycolmap.IncrementalPipelineOptions(
                    min_num_matches=15,
                    init_min_num_inliers=50,
                ),
            )
        except (AttributeError, TypeError):
            try:
                reconstructions = pycolmap.incremental_mapping(
                    database_path=str(db_path),
                    image_path=str(image_dir),
                    output_path=str(output_path),
                )
            except Exception as exc:
                raise RuntimeError(f"pycolmap.incremental_mapping failed: {exc}") from exc

        if not reconstructions:
            raise RuntimeError("SfM produced no reconstruction")

        # Pick the largest reconstruction (most registered images)
        recon = max(reconstructions.values(), key=lambda r: r.num_reg_images())
        num_registered = recon.num_reg_images()

        if callback:
            callback(0.70, f"SfM 完成，注册了 {num_registered}/{len(selected)} 帧")

        # 6. Extract 3D points
        points3d = recon.points3D
        if len(points3d) == 0:
            raise RuntimeError("No 3D points triangulated")

        # Strict filter: track length ≥3 and reprojection error < 3 px
        pts_list, col_list = [], []
        for pt in points3d.values():
            if pt.track.length() >= 3 and pt.error < 3.0:
                pts_list.append(pt.xyz)
                col_list.append(pt.color / 255.0)

        # Relaxed filter if strict yields too few
        if len(pts_list) < 50:
            pts_list, col_list = [], []
            for pt in points3d.values():
                if pt.track.length() >= 2:
                    pts_list.append(pt.xyz)
                    col_list.append(pt.color / 255.0)

        if len(pts_list) == 0:
            raise RuntimeError("No valid 3D points after filtering")

        points = np.array(pts_list, dtype=np.float32)
        colors = np.array(col_list, dtype=np.float32).clip(0, 1)

        return points, colors, num_registered

    except RuntimeError:
        raise  # propagate our own RuntimeErrors
    except Exception as exc:
        # Catch any unexpected pycolmap error and convert to RuntimeError
        raise RuntimeError(f"pycolmap SfM failed: {exc}") from exc
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tier 3 — OpenCV relative pose chaining
# ---------------------------------------------------------------------------

def _run_opencv_sfm(frames, callback, cancel_getter):
    """
    OpenCV-based relative pose chaining (no pycolmap dependency).

    Chains frame[i] → frame[i+1] poses and triangulates points at each step.

    Returns
    -------
    (points_3d : np.ndarray Nx3, colors : np.ndarray Nx3)

    Raises RuntimeError if the result contains fewer than 10 points.
    """
    selected = _select_frames(frames, max_frames=20, max_long_edge=640)
    if len(selected) < 2:
        raise RuntimeError("Not enough frames for OpenCV SfM")

    if callback:
        callback(0.15, f"使用 OpenCV 位姿估计（{len(selected)} 帧）...")

    h, w = selected[0].shape[:2]
    focal = max(h, w) * 1.2  # heuristic focal length
    K = np.array(
        [[focal, 0, w / 2],
         [0, focal, h / 2],
         [0,     0,     1]],
        dtype=np.float64,
    )

    sift = cv2.SIFT_create(nfeatures=4096)

    all_points = []
    all_colors = []

    # Prepare first frame
    img0_bgr = cv2.cvtColor(
        (selected[0] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    gray0 = cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2GRAY)
    kp0, des0 = sift.detectAndCompute(gray0, None)

    for i in range(1, len(selected)):
        if cancel_getter and cancel_getter():
            break

        img1_bgr = cv2.cvtColor(
            (selected[i] * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR
        )
        gray1 = cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(gray1, None)

        if des0 is None or des1 is None or len(des0) < 10 or len(des1) < 10:
            kp0, des0 = kp1, des1
            continue

        # Ratio-test matching
        bf = cv2.BFMatcher(cv2.NORM_L2)
        try:
            matches = bf.knnMatch(des0, des1, k=2)
        except Exception:
            kp0, des0 = kp1, des1
            continue

        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 15:
            kp0, des0 = kp1, des1
            continue

        pts0 = np.float32([kp0[m.queryIdx].pt for m in good])
        pts1 = np.float32([kp1[m.trainIdx].pt for m in good])

        # Essential matrix + recover pose
        E, mask_e = cv2.findEssentialMat(
            pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )
        if E is None or mask_e is None:
            kp0, des0 = kp1, des1
            continue

        inlier_mask = mask_e.ravel().astype(bool)
        if inlier_mask.sum() < 10:
            kp0, des0 = kp1, des1
            continue

        _, R, t, _ = cv2.recoverPose(
            E, pts0[inlier_mask], pts1[inlier_mask], K
        )

        # Triangulate between canonical camera (identity) and recovered pose
        P0 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = K @ np.hstack([R, t])

        pts0_in = pts0[inlier_mask].T  # 2×N
        pts1_in = pts1[inlier_mask].T  # 2×N

        pts4d = cv2.triangulatePoints(P0, P1, pts0_in, pts1_in)
        w_coord = pts4d[3:] + 1e-10
        pts3d = (pts4d[:3] / w_coord).T  # N×3

        # Depth / range filter
        valid = (
            (pts4d[3] > 0.01)
            & (pts3d[:, 2] > 0)
            & (np.abs(pts3d).max(axis=1) < 100)
        )
        pts3d = pts3d[valid]

        if len(pts3d) > 0:
            # Sample colors from frame 0 at the matched keypoint locations
            frame0 = selected[0]
            fh, fw = frame0.shape[:2]
            for pt_2d in pts0[inlier_mask][valid]:
                px = int(np.clip(pt_2d[0], 0, fw - 1))
                py = int(np.clip(pt_2d[1], 0, fh - 1))
                all_colors.append(frame0[py, px])
            all_points.extend(pts3d.tolist())

        # Advance window
        kp0, des0 = kp1, des1

        if callback:
            progress = 0.15 + (i / len(selected)) * 0.55
            callback(progress, f"OpenCV 位姿估计: {i}/{len(selected) - 1} 帧对")

    if len(all_points) < 10:
        raise RuntimeError(
            f"OpenCV SfM produced only {len(all_points)} points (need ≥10)"
        )

    points = np.array(all_points, dtype=np.float32)
    colors = np.array(all_colors, dtype=np.float32).clip(0, 1)
    return points, colors


# ---------------------------------------------------------------------------
# Tier 4 — Arc rotation fallback
# ---------------------------------------------------------------------------

def _arc_fallback(frames):
    """
    Generate a synthetic upper-hemisphere point cloud.

    This guarantees that the pipeline always returns *something* even when all
    real reconstruction tiers fail.  The colors are sampled from the first frame.

    Returns
    -------
    (points : np.ndarray Nx3, colors : np.ndarray Nx3)
    """
    n_points = 5000
    rng = np.random.default_rng(seed=42)

    theta = rng.uniform(0, np.pi, n_points)
    phi   = rng.uniform(0, 2 * np.pi, n_points)
    r     = rng.uniform(0.8, 1.2, n_points)

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.cos(theta)
    z = r * np.sin(theta) * np.sin(phi)

    # Keep upper hemisphere — roughly where hair lives
    keep = y > -0.2
    points = np.column_stack([x, y, z])[keep].astype(np.float32)

    if frames:
        frame = frames[0]
        fh, fw = frame.shape[:2]
        n = len(points)
        ys = rng.integers(0, fh, n)
        xs = rng.integers(0, fw, n)
        colors = frame[ys, xs].astype(np.float32)
    else:
        colors = rng.uniform(0.3, 0.6, (len(points), 3)).astype(np.float32)

    return points, colors


# ---------------------------------------------------------------------------
# Point cloud post-processing
# ---------------------------------------------------------------------------

def _filter_and_align_points(points, colors, hair_masks, frames):
    """
    Statistical outlier removal and optional subsampling.

    Points more than 3 standard deviations from the centroid are removed.
    The result is subsampled to at most 50 000 points.

    Parameters
    ----------
    points, colors : np.ndarray
    hair_masks     : list (not used in the current implementation but kept for
                     future projection-based filtering)
    frames         : list (same)

    Returns
    -------
    (points, colors) after filtering
    """
    if len(points) == 0:
        return points, colors

    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    threshold = dists.mean() + 3.0 * dists.std()
    valid = dists < threshold
    points = points[valid]
    colors = colors[valid]

    if len(points) > 50_000:
        idx = np.random.choice(len(points), 50_000, replace=False)
        points = points[idx]
        colors = colors[idx]

    return points, colors


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def reconstruct_from_frames(
    frames,
    hair_masks=None,
    callback=None,
    cancel_flag_getter=None,
):
    """
    Multi-view 3D reconstruction with 4-tier fallback ladder.

    Parameters
    ----------
    frames : list of np.ndarray
        HxWx3 float32 arrays in [0, 1].
    hair_masks : list of np.ndarray or None, optional
        HxW bool arrays, one per frame.  None entries are allowed.
    callback : callable, optional
        ``callback(progress: float, message: str)`` — progress in [0, 1].
    cancel_flag_getter : callable, optional
        Nullary callable; returns True to request cancellation.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
    colors : np.ndarray, shape (N, 3)
    tier   : int
        1 = full SfM, 2 = partial SfM, 3 = OpenCV chain, 4 = arc fallback.

    Raises
    ------
    RuntimeError
        Only if all tiers fail *and* the arc fallback also fails (essentially
        impossible — it is a pure synthetic generator).
    """
    if hair_masks is None:
        hair_masks = [None] * len(frames)

    if len(frames) < 2:
        logger.warning("reconstruct_from_frames: fewer than 2 frames — using arc fallback")
        pts, cols = _arc_fallback(frames)
        return pts, cols, 4

    # ------------------------------------------------------------------
    # Tier 1 & 2: pycolmap SfM
    # ------------------------------------------------------------------
    try:
        if callback:
            callback(0.05, "尝试 pycolmap 三维重建...")
        points, colors, num_registered = _run_pycolmap_sfm(
            frames, hair_masks, callback, cancel_flag_getter
        )
        points, colors = _filter_and_align_points(points, colors, hair_masks, frames)

        if num_registered >= 10 and len(points) >= 500:
            if callback:
                callback(0.85, f"SfM 完整重建成功: {len(points)} 个三维点")
            return points, colors, 1

        if num_registered >= 5 and len(points) >= 200:
            if callback:
                callback(
                    0.85,
                    f"SfM 部分重建: {num_registered} 帧, {len(points)} 点（质量受限）",
                )
            return points, colors, 2

        # Quality below tier-2 threshold — cascade to tier 3
        raise RuntimeError(
            f"SfM quality too low: {num_registered} frames, {len(points)} points"
        )

    except Exception as exc:
        logger.info("pycolmap tier failed (%s: %s) — falling back to OpenCV", type(exc).__name__, exc)
        if callback:
            callback(
                0.10,
                f"pycolmap 失败 ({type(exc).__name__})，切换 OpenCV 位姿估计...",
            )

    # ------------------------------------------------------------------
    # Tier 3: OpenCV relative pose chaining
    # ------------------------------------------------------------------
    try:
        points, colors = _run_opencv_sfm(frames, callback, cancel_flag_getter)
        points, colors = _filter_and_align_points(points, colors, hair_masks, frames)
        if callback:
            callback(0.85, f"OpenCV 位姿估计完成: {len(points)} 个三维点")
        return points, colors, 3

    except Exception as exc:
        logger.info("OpenCV SfM tier failed (%s: %s) — using arc fallback", type(exc).__name__, exc)
        if callback:
            callback(
                0.10,
                f"OpenCV SfM 失败 ({type(exc).__name__})，使用弧形占位符...",
            )

    # ------------------------------------------------------------------
    # Tier 4: Arc fallback (always succeeds)
    # ------------------------------------------------------------------
    points, colors = _arc_fallback(frames)
    if callback:
        callback(
            0.85,
            f"使用弧形占位符点云 ({len(points)} 点)，建议使用更好的视频素材",
        )
    return points, colors, 4
