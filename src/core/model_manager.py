"""
Model manager: checks local HuggingFace cache and provides download progress callbacks.

HuggingFace mirror
------------------
If the environment variable HF_ENDPOINT is set, the HuggingFace libraries
automatically use that endpoint.  Users can also configure a mirror via
Settings → AI 模型 → 镜像地址 (stored in settings.json as "hf_endpoint").

Popular mirrors for China:
  https://hf-mirror.com
"""
import logging
import os
import threading
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def apply_hf_mirror():
    """
    Apply the HuggingFace endpoint from settings (or HF_ENDPOINT env var).
    Call once at startup or before any model download.
    """
    # Env var takes priority (allows CI/CD or advanced users to override)
    if os.environ.get('HF_ENDPOINT'):
        return

    try:
        from src.config.settings_manager import load_settings
        endpoint = load_settings().get('hf_endpoint', '').strip()
        if endpoint:
            os.environ['HF_ENDPOINT'] = endpoint
    except Exception:
        pass


def _summarise_download_error(exc: Exception) -> str:
    """
    Translate a raw HF / requests / OS exception into a short, actionable
    message suitable for displaying in a small dialog.  The full traceback
    is always logged separately.
    """
    s = f"{type(exc).__name__}: {exc}"
    low = s.lower()

    # Connectivity / DNS / firewall — almost always means HF is blocked or slow
    if any(token in low for token in (
        "max retries", "connection", "timeout", "name or service",
        "failed to resolve", "getaddrinfo", "name resolution",
        "newconnectionerror", "remotedisconnected", "connectionreseterror",
    )):
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        if "huggingface.co" in endpoint:
            return (
                "网络无法连接 HuggingFace 官方源。\n"
                "建议：Settings → AI 模型 → 镜像地址填 https://hf-mirror.com"
            )
        return f"网络连接失败（当前端点：{endpoint}）。请检查网络或更换镜像。"

    # SSL
    if "ssl" in low or "certificate" in low:
        return "SSL 证书验证失败。建议升级 certifi：pip install -U certifi"

    # 401/403/private models
    if "401" in s or "unauthorized" in low or "403" in s:
        return "访问被拒绝（401/403）。模型可能需要登录，或镜像源已过期。"

    # 404 — model not found at endpoint
    if "404" in s or "not found" in low or "repository not found" in low:
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
        return f"模型在 {endpoint} 找不到。镜像可能不包含此模型，请换源或用官方源。"

    # Disk space
    if "no space" in low or "disk full" in low:
        return "磁盘空间不足。请清理空间后重试。"

    # Permission
    if "permission" in low or "errno 13" in low:
        return "权限不足。请检查 ~/.cache/huggingface 是否可写。"

    # transformers missing
    if "no module named" in low and "transformers" in low:
        return "transformers 未安装，请运行 pip install -r requirements.txt"

    # Fallback: return the short exception text (truncated)
    return s if len(s) <= 160 else s[:160] + "…"

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
    # Apply mirror before any HF network call
    apply_hf_mirror()

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
                logger.info(
                    "Downloading %s from endpoint=%s",
                    repo_id, os.environ.get("HF_ENDPOINT", "https://huggingface.co"),
                )
                model_info["loader"](repo_id)
            finally:
                _sim_stop.set()
                sim_thread.join(timeout=2.0)

            if progress_callback:
                progress_callback(progress_base + weight, f"✓ {name} 下载完成")
            logger.info("Successfully downloaded %s", repo_id)

        except Exception as e:
            # Full traceback to the log file / log window
            logger.exception("Download of %s failed", repo_id)
            short = _summarise_download_error(e)
            if progress_callback:
                # Show a friendly summary in the dialog; full details are in the log
                progress_callback(progress_base, f"✗ {name} 下载失败：{short}")
            return False

        progress_base += weight

    if progress_callback:
        progress_callback(1.0, "所有模型已就绪")
    return True
