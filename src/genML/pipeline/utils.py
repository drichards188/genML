"""Utility helpers shared across pipeline stages."""

from __future__ import annotations

import gc
import logging

from src.genML.gpu_utils import get_gpu_memory_usage, is_cuml_available
from src.genML.pipeline import config
from src.genML.pipeline.optional_dependencies import CATBOOST_AVAILABLE, cb

logger = logging.getLogger(__name__)


def cleanup_memory(stage_name: str = "", aggressive: bool = False) -> None:
    """Force garbage collection and GPU memory cleanup to prevent leaks."""
    logger.info("[Cleanup] Starting cleanup for: %s (aggressive=%s)", stage_name, aggressive)

    mem_before = None
    if is_cuml_available():
        mem_before = get_gpu_memory_usage()
        if mem_before:
            logger.info("[Cleanup] GPU memory before cleanup: %.2fGB", mem_before)

    if config.MEMORY_CONFIG["enable_gc_between_trials"]:
        if aggressive:
            logger.info("[Cleanup] Running aggressive garbage collection (3 passes)")
            for idx in range(3):
                collected = gc.collect()
                logger.info("[Cleanup] GC pass %s collected %s objects", idx + 1, collected)
        else:
            logger.info("[Cleanup] Running standard garbage collection")
            collected = gc.collect()
            logger.info("[Cleanup] GC collected %s objects", collected)

    if config.MEMORY_CONFIG["enable_gpu_memory_cleanup"] and is_cuml_available():
        try:
            import cupy as cp  # type: ignore

            logger.info("[Cleanup] Freeing GPU memory pools (cuML)")
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

            mem_after = get_gpu_memory_usage()
            if mem_after:
                logger.info("[Cleanup] GPU memory after cleanup: %.2fGB", mem_after)
                if mem_before and mem_after < mem_before:
                    logger.info("[Cleanup] Freed %.2fGB of GPU memory", mem_before - mem_after)
        except Exception as exc:  # pragma: no cover - defensive cleanup
            logger.warning("[Cleanup] GPU memory cleanup failed: %s", exc)

    if CATBOOST_AVAILABLE:
        try:
            clear_cache = getattr(getattr(cb, "_catboost", None), "clear_cache", None)
            if clear_cache:
                logger.info("[Cleanup] Clearing CatBoost internal cache")
                clear_cache()
        except Exception as exc:
            logger.warning("[Cleanup] CatBoost cache cleanup failed: %s", exc)
