"""
Model manager: checks local HuggingFace cache and provides download progress callbacks.
"""
import os
import threading
from pathlib import Path
from typing import Callable, Optional

# The two models used by the app
MODELS = {
    "seg": {
        "repo_id": "jonathandinu/face-parsing",
        "display_name": "发丝分割模型 (face-parsing)",
        "approx_size_mb": 85,
    },
    "depth": {
        "repo_id": "depth-anything/Depth-Anything-V2-Small-hf",
        "display_name": "深度估计模型 (Depth-Anything-V2)",
        "approx_size_mb": 99,
    },
}


def get_hf_cache_dir() -> Path:
    """Returns the HuggingFace cache directory."""
    hf_home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    if hf_home:
        return Path(hf_home)
    return Path.home() / ".cache" / "huggingface" / "hub"


def is_model_cached(repo_id: str) -> bool:
    """
    Check if a model is already in the local HuggingFace cache.
    Looks for the model directory in the standard HF cache structure.
    """
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            if repo.repo_id == repo_id:
                return True
        return False
    except Exception:
        # Fallback: check directory existence
        cache_dir = get_hf_cache_dir()
        # HF stores models as "models--org--name"
        folder_name = "models--" + repo_id.replace("/", "--")
        return (cache_dir / folder_name).exists()


def get_models_to_download() -> list:
    """
    Returns list of model dicts that are NOT yet cached locally.
    Each dict has: repo_id, display_name, approx_size_mb
    """
    try:
        from transformers import AutoConfig  # noqa: F401 — just check importability
    except ImportError:
        return []  # transformers not installed, skip

    missing = []
    for key, info in MODELS.items():
        if not is_model_cached(info["repo_id"]):
            missing.append(info.copy())
    return missing


def get_all_models_status() -> list:
    """Returns all models with their cache status."""
    result = []
    for key, info in MODELS.items():
        status = is_model_cached(info["repo_id"])
        entry = info.copy()
        entry["cached"] = status
        result.append(entry)
    return result


def download_models(
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_flag_getter: Optional[Callable[[], bool]] = None,
) -> bool:
    """
    Download all missing models from HuggingFace.

    progress_callback(progress: float [0.0-1.0], message: str)
    cancel_flag_getter() -> bool: return True to cancel

    Returns True if all models downloaded successfully, False on error/cancel.
    """
    try:
        from transformers import (
            SegformerImageProcessor, SegformerForSemanticSegmentation,
            AutoImageProcessor, AutoModelForDepthEstimation,
        )
    except ImportError:
        if progress_callback:
            progress_callback(0.0, "错误: transformers 库未安装，请运行 pip install transformers")
        return False

    models_to_load = [
        {
            "name": MODELS["seg"]["display_name"],
            "repo_id": MODELS["seg"]["repo_id"],
            "approx_size_mb": MODELS["seg"]["approx_size_mb"],
            "loader": lambda rid: (
                SegformerImageProcessor.from_pretrained(rid),
                SegformerForSemanticSegmentation.from_pretrained(rid),
            ),
            "weight": 0.46,  # Proportion of total download (85/184)
        },
        {
            "name": MODELS["depth"]["display_name"],
            "repo_id": MODELS["depth"]["repo_id"],
            "approx_size_mb": MODELS["depth"]["approx_size_mb"],
            "loader": lambda rid: (
                AutoImageProcessor.from_pretrained(rid),
                AutoModelForDepthEstimation.from_pretrained(rid),
            ),
            "weight": 0.54,  # Proportion of total download (99/184)
        },
    ]

    progress_base = 0.0

    for model_info in models_to_load:
        if cancel_flag_getter and cancel_flag_getter():
            return False

        name = model_info["name"]
        repo_id = model_info["repo_id"]
        weight = model_info["weight"]

        # Skip if already cached
        if is_model_cached(repo_id):
            if progress_callback:
                progress_callback(progress_base + weight, f"✓ {name} (已缓存)")
            progress_base += weight
            continue

        if progress_callback:
            progress_callback(progress_base, f"正在下载 {name}...")

        try:
            # Start a lightweight progress simulation thread while the actual
            # download (blocking) runs in the current thread.
            _sim_stop = threading.Event()

            def _progress_sim(pb=progress_base, w=weight, n=name, stop=_sim_stop):
                import time
                steps = 40
                for step in range(steps):
                    if stop.is_set() or (cancel_flag_getter and cancel_flag_getter()):
                        return
                    frac = (step + 1) / steps
                    if progress_callback:
                        progress_callback(
                            pb + w * frac * 0.9,
                            f"正在下载 {n}... ({frac * 100:.0f}%)"
                        )
                    time.sleep(0.5)

            sim_thread = threading.Thread(target=_progress_sim, daemon=True)
            sim_thread.start()

            try:
                # Actual download (blocking) — uses HF transformers cache
                model_info["loader"](repo_id)
            finally:
                _sim_stop.set()
                sim_thread.join(timeout=2.0)

            if progress_callback:
                progress_callback(progress_base + weight, f"✓ {name} 下载完成")

        except Exception as e:
            if progress_callback:
                progress_callback(progress_base, f"✗ {name} 下载失败: {e}")
            return False

        progress_base += weight

    if progress_callback:
        progress_callback(1.0, "所有模型已就绪")
    return True
