"""
Batch Processor Module
======================

Queue multiple image-sets and process them sequentially without
manual intervention.  Each job:
  1. Loads a set of ≥3 images
  2. Generates a GaussianCloud
  3. Extracts hair strands
  4. Saves a .ghc project file
  5. Optionally exports FBX / GLB
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Callable, List, Optional


class JobStatus(Enum):
    PENDING    = auto()
    PROCESSING = auto()
    DONE       = auto()
    FAILED     = auto()
    CANCELLED  = auto()


@dataclass
class BatchJob:
    image_paths:  List[str]     = field(default_factory=list)
    output_dir:   str           = ""
    auto_export:  Optional[str] = None   # 'fbx' | 'glb' | None
    id:           str           = field(default_factory=lambda: uuid.uuid4().hex[:8])
    status:       JobStatus     = JobStatus.PENDING
    result_path:  Optional[str] = None
    error:        Optional[str] = None
    num_splats:   int           = 0
    num_strands:  int           = 0

    @property
    def label(self) -> str:
        n = len(self.image_paths)
        base = Path(self.image_paths[0]).stem if self.image_paths else "job"
        return f"{base} ({n} 张图)"


class BatchProcessor:
    """
    Sequential batch processor that re-uses pre-loaded ML models.

    Usage
    -----
    processor = BatchProcessor(generator, extractor)
    processor.add_job(image_paths, output_dir, auto_export='fbx')
    ...
    threading.Thread(target=processor.run_all, kwargs={...}).start()
    """

    def __init__(self, generator, extractor):
        self._generator = generator
        self._extractor = extractor
        self._jobs: List[BatchJob] = []
        self.is_running = False
        self._cancel_flag = False

    # ------------------------------------------------------------------
    # Job management
    # ------------------------------------------------------------------

    def add_job(
        self,
        image_paths: List[str],
        output_dir: str,
        auto_export: Optional[str] = None,
    ) -> BatchJob:
        job = BatchJob(
            image_paths=image_paths,
            output_dir=output_dir,
            auto_export=auto_export,
        )
        self._jobs.append(job)
        return job

    def remove_job(self, job_id: str):
        self._jobs = [j for j in self._jobs if j.id != job_id]

    def clear_finished(self):
        self._jobs = [
            j for j in self._jobs
            if j.status in (JobStatus.PENDING, JobStatus.PROCESSING)
        ]

    @property
    def jobs(self) -> List[BatchJob]:
        return list(self._jobs)

    @property
    def pending_count(self) -> int:
        return sum(1 for j in self._jobs if j.status == JobStatus.PENDING)

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def cancel(self):
        self._cancel_flag = True
        self._generator.cancel()

    def run_all(
        self,
        on_job_update: Optional[Callable[["BatchJob"], None]] = None,
        on_progress:   Optional[Callable[[int, int, float, str], None]] = None,
    ):
        """
        Process every PENDING job in order.  Designed to run in a daemon
        thread; main-thread callbacks are the caller's responsibility.

        on_job_update(job)
        on_progress(job_idx, total_jobs, step_progress 0-1, step_message)
        """
        self.is_running = True
        self._cancel_flag = False

        pending = [j for j in self._jobs if j.status == JobStatus.PENDING]
        total = len(pending)

        for idx, job in enumerate(pending):
            if self._cancel_flag:
                job.status = JobStatus.CANCELLED
                if on_job_update:
                    on_job_update(job)
                break

            job.status = JobStatus.PROCESSING
            if on_job_update:
                on_job_update(job)

            try:
                self._process_job(job, idx, total, on_progress)
                job.status = JobStatus.DONE
            except Exception as exc:
                job.status = JobStatus.FAILED
                job.error = str(exc)

            if on_job_update:
                on_job_update(job)

        self.is_running = False

    def _process_job(
        self,
        job: BatchJob,
        idx: int,
        total: int,
        on_progress: Optional[Callable],
    ):
        import numpy as np
        from PIL import Image as PILImage

        def cb(prog: float, msg: str):
            if on_progress:
                on_progress(idx, total, prog, msg)

        # ---- Load images ----
        cb(0.0, f"[{idx+1}/{total}] 加载图片…")
        images = []
        for p in job.image_paths:
            try:
                img = np.array(PILImage.open(p).convert("RGB"))
                images.append(img)
            except Exception:
                pass

        if len(images) < 3:
            raise ValueError(f"只加载到 {len(images)} 张有效图片（需要 ≥3）")

        # ---- Generate Gaussians ----
        def gen_cb(prog, msg):
            cb(prog * 0.6, f"[{idx+1}/{total}] {msg}")

        cloud = self._generator.generate_from_images(images, callback=gen_cb)
        job.num_splats = cloud.num_splats

        if self._cancel_flag:
            raise RuntimeError("已取消")

        # ---- Extract strands ----
        def ext_cb(prog, msg):
            cb(0.6 + prog * 0.3, f"[{idx+1}/{total}] {msg}")

        strands = self._extractor.extract(cloud, callback=ext_cb)
        job.num_strands = strands.num_strands

        # ---- Save .ghc ----
        cb(0.92, f"[{idx+1}/{total}] 保存项目文件…")
        out_dir = Path(job.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(job.image_paths[0]).stem if job.image_paths else "result"
        ghc_path = str(out_dir / f"{base}_hair.ghc")

        from src.core.project_io import save_project
        if not save_project(ghc_path, cloud, strands, image_paths=job.image_paths):
            raise RuntimeError("保存 .ghc 文件失败")
        job.result_path = ghc_path

        # ---- Auto export ----
        if job.auto_export == "fbx" and strands.num_strands > 0:
            cb(0.95, f"[{idx+1}/{total}] 导出 FBX…")
            from src.export.fbx_exporter import FBXExporter, FBXExportOptions
            fbx_path = str(out_dir / f"{base}_hair.fbx")
            FBXExporter(FBXExportOptions(scale_factor=25.0)).export(strands, fbx_path)

        elif job.auto_export == "glb" and strands.num_strands > 0:
            cb(0.95, f"[{idx+1}/{total}] 导出 GLB…")
            from src.export.glb_exporter import GLBExporter, GLBExportOptions
            glb_path = str(out_dir / f"{base}_hair.glb")
            GLBExporter(GLBExportOptions(scale_factor=0.25, up_axis="Z")).export_strands(
                strands, glb_path
            )

        cb(1.0, f"[{idx+1}/{total}] 完成 — {cloud.num_splats} 个高斯，{strands.num_strands} 条发丝")
