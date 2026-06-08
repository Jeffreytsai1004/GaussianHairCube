"""
Project I/O Module
==================

Save and load GaussianHairCube project files (.ghc).

Format: ZIP archive containing
  meta.json       – version, timestamps, counts, image paths
  gaussians.npz   – positions, colors, scales, opacities, rotations, bounds
  strands.npz     – packed strand data (points_flat, strand_lengths, radii, colors)
"""

import io
import json
import logging
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np

from src.core.gaussian_generator import GaussianCloud, GaussianSplat
from src.core.hair_strands import HairStrand, HairStrandCollection

logger = logging.getLogger(__name__)

_FORMAT_VERSION = "1.0"
_GHC_SUFFIX = ".ghc"


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_project(
    path: str,
    cloud: Optional[GaussianCloud],
    strands: Optional[HairStrandCollection],
    image_paths: Optional[List[str]] = None,
) -> bool:
    """
    Save the current session to a .ghc file.

    Parameters
    ----------
    path        : Destination path (will be created/overwritten).
    cloud       : Gaussian cloud, or None to omit.
    strands     : Hair strand collection, or None to omit.
    image_paths : List of input image paths stored as metadata (reference only).

    Returns True on success.
    """
    try:
        meta = {
            "version": _FORMAT_VERSION,
            "created": datetime.now().isoformat(timespec="seconds"),
            "num_splats": cloud.num_splats if cloud else 0,
            "num_strands": strands.num_strands if strands else 0,
            "image_paths": image_paths or [],
        }

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("meta.json", json.dumps(meta, indent=2, ensure_ascii=False))

            if cloud is not None and cloud.num_splats > 0:
                zf.writestr("gaussians.npz", _cloud_to_bytes(cloud))

            if strands is not None and strands.num_strands > 0:
                zf.writestr("strands.npz", _strands_to_bytes(strands))

        return True
    except Exception as exc:
        logger.exception("save_project failed: %s", exc)
        return False


def _cloud_to_bytes(cloud: GaussianCloud) -> bytes:
    positions  = np.stack([s.position  for s in cloud.splats], axis=0).astype(np.float32)
    colors     = np.stack([s.color     for s in cloud.splats], axis=0).astype(np.float32)
    scales     = np.stack([s.scale     for s in cloud.splats], axis=0).astype(np.float32)
    opacities  = np.array([s.opacity   for s in cloud.splats], dtype=np.float32)
    rotations  = np.stack([s.rotation  for s in cloud.splats], axis=0).astype(np.float32)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        positions=positions, colors=colors, scales=scales,
        opacities=opacities, rotations=rotations,
        bounds_min=cloud.bounds_min.astype(np.float32),
        bounds_max=cloud.bounds_max.astype(np.float32),
    )
    return buf.getvalue()


def _strands_to_bytes(strands: HairStrandCollection) -> bytes:
    lengths     = np.array([s.num_points for s in strands.strands], dtype=np.int32)
    pts_flat    = np.vstack([s.points    for s in strands.strands]).astype(np.float32)
    radii_flat  = np.concatenate([s.radii  for s in strands.strands]).astype(np.float32)
    colors_flat = np.vstack([s.colors   for s in strands.strands]).astype(np.float32)

    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        lengths=lengths,
        points_flat=pts_flat,
        radii_flat=radii_flat,
        colors_flat=colors_flat,
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_project(path: str):
    """
    Load a .ghc project file.

    Returns
    -------
    (cloud, strands, meta)
        cloud   : GaussianCloud or None
        strands : HairStrandCollection or None
        meta    : dict with version, created, num_splats, num_strands, image_paths
    """
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()

            # Meta
            meta = json.loads(zf.read("meta.json").decode("utf-8"))

            # Gaussians
            cloud = None
            if "gaussians.npz" in names:
                buf = io.BytesIO(zf.read("gaussians.npz"))
                cloud = _bytes_to_cloud(np.load(buf))

            # Strands
            strand_col = None
            if "strands.npz" in names:
                buf = io.BytesIO(zf.read("strands.npz"))
                strand_col = _bytes_to_strands(np.load(buf))

        return cloud, strand_col, meta
    except Exception as exc:
        logger.exception("load_project failed: %s", exc)
        return None, None, {}


def _bytes_to_cloud(npz) -> GaussianCloud:
    positions = npz["positions"]   # (N, 3)
    colors    = npz["colors"]      # (N, 3)
    scales    = npz["scales"]      # (N, 3)
    opacities = npz["opacities"]   # (N,)
    rotations = npz["rotations"]   # (N, 4)

    splats = [
        GaussianSplat(
            position=positions[i],
            covariance=np.diag(scales[i] ** 2),
            color=colors[i],
            opacity=float(opacities[i]),
            scale=scales[i],
            rotation=rotations[i],
        )
        for i in range(len(positions))
    ]
    return GaussianCloud(
        splats=splats,
        bounds_min=npz["bounds_min"],
        bounds_max=npz["bounds_max"],
    )


def _bytes_to_strands(npz) -> HairStrandCollection:
    lengths     = npz["lengths"]       # (K,)
    pts_flat    = npz["points_flat"]   # (M, 3)
    radii_flat  = npz["radii_flat"]    # (M,)
    colors_flat = npz["colors_flat"]   # (M, 3)

    strands = []
    offset = 0
    for length in lengths:
        length = int(length)
        strand = HairStrand(
            points=pts_flat[offset:offset + length],
            radii=radii_flat[offset:offset + length],
            colors=colors_flat[offset:offset + length],
        )
        strands.append(strand)
        offset += length

    return HairStrandCollection(strands=strands)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_ghc_file(path: str) -> bool:
    return Path(path).suffix.lower() == _GHC_SUFFIX
