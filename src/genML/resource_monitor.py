"""
Real-time resource monitoring for ML Pipeline.

This module provides background monitoring of system resources (CPU, RAM, GPU)
during pipeline execution. Metrics are periodically updated to the progress tracker.
"""

import logging
import psutil
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """
    Background thread that monitors system resource usage.

    Periodically samples CPU, RAM, and GPU metrics and updates the progress tracker.
    """

    def __init__(self, progress_tracker, interval_seconds: float = 2.0):
        """
        Initialize resource monitor.

        Args:
            progress_tracker: ProgressTracker instance to update with metrics
            interval_seconds: How often to sample resources (default: 2 seconds)
        """
        self.progress_tracker = progress_tracker
        self.interval_seconds = interval_seconds
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Try to import GPU utilities
        try:
            from src.genML.gpu_utils import get_gpu_memory_info
            self._get_gpu_memory = get_gpu_memory_info
            self._gpu_available = True
        except Exception as e:
            logger.debug(f"GPU monitoring not available: {e}")
            self._get_gpu_memory = None
            self._gpu_available = False

        logger.info(f"ResourceMonitor initialized (GPU monitoring: {self._gpu_available})")

    def _monitor_loop(self):
        """Main monitoring loop (runs in background thread)."""
        logger.info("Resource monitoring started")

        while not self._stop_event.is_set():
            try:
                # Get CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Get RAM usage
                ram = psutil.virtual_memory()
                ram_mb = ram.used / (1024 * 1024)

                # Get GPU usage if available
                gpu_memory_mb = 0
                gpu_memory_total_mb = 0

                if self._gpu_available and self._get_gpu_memory:
                    try:
                        gpu_info = self._get_gpu_memory()
                        if gpu_info:
                            gpu_memory_mb = gpu_info.get('used_mb', 0)
                            gpu_memory_total_mb = gpu_info.get('total_mb', 0)
                    except Exception as e:
                        logger.debug(f"Failed to get GPU memory: {e}")

                # Update progress tracker
                self.progress_tracker.track_resources(
                    gpu_memory_mb=gpu_memory_mb,
                    gpu_memory_total_mb=gpu_memory_total_mb,
                    cpu_percent=cpu_percent,
                    ram_mb=ram_mb
                )

            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")

            # Wait for next interval or stop signal
            self._stop_event.wait(self.interval_seconds)

        logger.info("Resource monitoring stopped")

    def start(self):
        """Start background resource monitoring."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Resource monitor already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True, name="ResourceMonitor")
        self._thread.start()
        logger.info("Resource monitor thread started")

    def stop(self):
        """Stop background resource monitoring."""
        if self._thread is None or not self._thread.is_alive():
            logger.warning("Resource monitor not running")
            return

        logger.info("Stopping resource monitor...")
        self._stop_event.set()
        self._thread.join(timeout=5.0)

        if self._thread.is_alive():
            logger.warning("Resource monitor thread did not stop cleanly")
        else:
            logger.info("Resource monitor stopped successfully")

    def __enter__(self):
        """Context manager entry - start monitoring."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - stop monitoring."""
        self.stop()
        return False
