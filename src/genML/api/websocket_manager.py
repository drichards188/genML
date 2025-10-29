"""
WebSocket manager for real-time progress updates.

This module manages WebSocket connections and broadcasts progress updates
to connected clients by watching the progress file for changes.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Set, Optional
from fastapi import WebSocket
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

logger = logging.getLogger(__name__)


class ProgressFileWatcher(FileSystemEventHandler):
    """
    Watches progress file for changes and triggers callbacks.
    """

    def __init__(self, progress_file: Path, on_change_callback, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Initialize file watcher.

        Args:
            progress_file: Path to progress JSON file
            on_change_callback: Async callback to call when file changes
            loop: Event loop to schedule callbacks in
        """
        self.progress_file = progress_file.resolve()
        self.on_change_callback = on_change_callback
        self._last_modified = 0
        self.loop = loop

    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            # Check if it's our progress file
            if Path(event.src_path).resolve() == self.progress_file:
                # Debounce rapid file changes
                import time
                current_time = time.time()
                if current_time - self._last_modified > 0.05:  # 50ms debounce
                    self._last_modified = current_time
                    # Schedule the callback in the main event loop
                    if self.loop and not self.loop.is_closed():
                        asyncio.run_coroutine_threadsafe(self.on_change_callback(), self.loop)


class WebSocketManager:
    """
    Manages WebSocket connections and broadcasts progress updates.
    """

    def __init__(self, progress_file: Path):
        """
        Initialize WebSocket manager.

        Args:
            progress_file: Path to progress JSON file to watch
        """
        self.progress_file = progress_file.resolve()
        self.active_connections: Set[WebSocket] = set()
        self._observer: Observer = None
        self._watcher: ProgressFileWatcher = None
        self._polling_task: Optional[asyncio.Task] = None
        self._poll_interval = 1.0  # seconds
        self._last_payload = None
        self._last_mtime: Optional[float] = None

        logger.info(f"WebSocketManager initialized (watching {progress_file})")

    async def connect(self, websocket: WebSocket):
        """
        Accept and register a new WebSocket connection.

        Args:
            websocket: WebSocket connection to register
        """
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")

        # Send current progress immediately on connection
        await self._send_current_progress(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Unregister a WebSocket connection.

        Args:
            websocket: WebSocket connection to remove
        """
        self.active_connections.discard(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def _send_current_progress(self, websocket: WebSocket):
        """
        Send current progress to a specific websocket.

        Args:
            websocket: WebSocket to send progress to
        """
        try:
            if self.progress_file.exists():
                try:
                    with open(self.progress_file, 'r') as f:
                        content = f.read()
                        if not content or content.isspace():
                            # File is empty, send idle status
                            await websocket.send_json({
                                "status": "idle",
                                "message": "No active pipeline run"
                            })
                            return
                        progress_data = json.loads(content)
                except json.JSONDecodeError as je:
                    # File has invalid JSON, likely being written - send idle for now
                    logger.debug(f"Progress file has invalid JSON on initial connect: {je}")
                    await websocket.send_json({
                        "status": "idle",
                        "message": "Pipeline starting..."
                    })
                    return

                self._last_payload = progress_data
                try:
                    self._last_mtime = self.progress_file.stat().st_mtime
                except FileNotFoundError:
                    self._last_mtime = None
                await websocket.send_json(progress_data)
            else:
                # Send empty progress if file doesn't exist
                await websocket.send_json({
                    "status": "idle",
                    "message": "No active pipeline run"
                })
        except Exception as e:
            logger.error(f"Failed to send current progress: {e}")

    async def broadcast_progress(self):
        """
        Broadcast current progress to all connected clients.
        """
        if not self.active_connections:
            return

        try:
            if self.progress_file.exists():
                # Read file with error handling for concurrent writes
                try:
                    with open(self.progress_file, 'r') as f:
                        content = f.read()
                        if not content or content.isspace():
                            # File is empty or being written, skip this update
                            logger.debug("Progress file is empty, skipping broadcast")
                            return
                        progress_data = json.loads(content)
                except json.JSONDecodeError as je:
                    # File is being written to, skip this update
                    logger.debug(f"Progress file has invalid JSON (likely being written): {je}")
                    return
                except Exception as read_error:
                    logger.warning(f"Error reading progress file: {read_error}")
                    return

                if self._last_payload == progress_data:
                    return
                self._last_payload = progress_data
                try:
                    self._last_mtime = self.progress_file.stat().st_mtime
                except FileNotFoundError:
                    self._last_mtime = None

                # Send to all connected clients
                disconnected = set()
                for websocket in self.active_connections:
                    try:
                        await websocket.send_json(progress_data)
                    except Exception as e:
                        logger.warning(f"Failed to send to websocket: {e}")
                        disconnected.add(websocket)

                # Remove disconnected clients
                for websocket in disconnected:
                    self.disconnect(websocket)

                logger.debug(f"Progress broadcast to {len(self.active_connections)} clients")

        except Exception as e:
            logger.error(f"Failed to broadcast progress: {e}")

    def start_watching(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """
        Start watching the progress file for changes.

        Args:
            loop: Event loop to use for scheduling async callbacks from the file watcher thread
        """
        if self._observer is not None:
            logger.warning("File watcher already running")
            return

        # Ensure progress directory exists
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        # Get the event loop if not provided
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                logger.warning("No running event loop found - file watcher may not work correctly")

        # Create file watcher with event loop
        self._watcher = ProgressFileWatcher(
            self.progress_file,
            self.broadcast_progress,
            loop=loop
        )

        # Set up observer
        self._observer = Observer()
        self._observer.schedule(
            self._watcher,
            str(self.progress_file.parent),
            recursive=False
        )
        self._observer.start()

        logger.info("Progress file watcher started")

        # Capture initial mtime
        try:
            if self.progress_file.exists():
                self._last_mtime = self.progress_file.stat().st_mtime
        except FileNotFoundError:
            self._last_mtime = None

        # Start polling fallback to ensure updates even if watchdog misses them
        if loop and not loop.is_closed():
            self._polling_task = loop.create_task(self._poll_progress())
            logger.info("Progress polling task started")

    def stop_watching(self):
        """Stop watching the progress file."""
        if self._polling_task is not None:
            loop = self._polling_task.get_loop()
            self._polling_task.cancel()
            if loop.is_running():
                loop.call_soon_threadsafe(lambda: None)
            self._polling_task = None
            logger.info("Progress polling task stopped")

        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=2.0)
            self._observer = None
            self._watcher = None
            logger.info("Progress file watcher stopped")

    async def close_all(self):
        """Close all active WebSocket connections."""
        for websocket in list(self.active_connections):
            try:
                await websocket.close()
            except Exception as e:
                logger.debug(f"Error closing websocket: {e}")
        self.active_connections.clear()
        logger.info("All WebSocket connections closed")

    async def _poll_progress(self):
        """Periodic polling fallback to catch updates missed by watchdog."""
        try:
            while True:
                try:
                    if self.progress_file.exists():
                        mtime = self.progress_file.stat().st_mtime
                        if self._last_mtime is None or mtime > self._last_mtime + 1e-6:
                            self._last_mtime = mtime
                            await self.broadcast_progress()
                except FileNotFoundError:
                    self._last_mtime = None
                except Exception as exc:
                    logger.debug(f"Progress polling error: {exc}")

                await asyncio.sleep(self._poll_interval)
        except asyncio.CancelledError:
            logger.debug("Progress polling task cancelled")
            raise
