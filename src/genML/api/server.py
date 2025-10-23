"""
FastAPI server for ML Pipeline Dashboard.

Provides REST API endpoints and WebSocket support for real-time monitoring
of the ML pipeline execution.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured application instance
    """
    app = FastAPI(
        title="ML Pipeline Dashboard API",
        description="Real-time monitoring API for GenML Pipeline",
        version="1.0.0"
    )

    # Enable CORS for React frontend
    allowed_origins = [
        "http://localhost",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        "http://0.0.0.0:5173",
        "http://0.0.0.0:8000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialize WebSocket manager
    progress_file = Path("outputs/progress/current_run.json")
    ws_manager = WebSocketManager(progress_file)

    # Store manager in app state
    app.state.ws_manager = ws_manager

    # ========================================================================
    # API Endpoints
    # ========================================================================

    @app.get("/api/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": "ml-pipeline-api"}

    @app.get("/api/status")
    async def get_status():
        """Get current pipeline status."""
        try:
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    return json.load(f)
            else:
                return {
                    "status": "idle",
                    "message": "No active pipeline run",
                    "run_id": None
                }
        except Exception as e:
            logger.error(f"Failed to read progress file: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/runs")
    async def get_runs():
        """Get list of all pipeline runs."""
        try:
            archive_dir = Path("outputs/progress/archive")
            if not archive_dir.exists():
                return []

            runs = []
            for run_file in sorted(archive_dir.glob("*.json"), reverse=True):
                try:
                    with open(run_file, 'r') as f:
                        data = json.load(f)
                        runs.append({
                            "run_id": data.get("run_id"),
                            "dataset_name": data.get("dataset_name"),
                            "status": data.get("status"),
                            "started_at": data.get("started_at"),
                            "completed_at": data.get("completed_at"),
                            "duration_seconds": data.get("resources", {}).get("elapsed_seconds")
                        })
                except Exception as e:
                    logger.warning(f"Failed to read run file {run_file}: {e}")
                    continue

            return runs
        except Exception as e:
            logger.error(f"Failed to list runs: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/runs/{run_id}")
    async def get_run(run_id: str):
        """Get detailed information for a specific run."""
        try:
            # Check current run first
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    data = json.load(f)
                    if data.get("run_id") == run_id:
                        return data

            # Check archive
            archive_file = Path(f"outputs/progress/archive/{run_id}.json")
            if archive_file.exists():
                with open(archive_file, 'r') as f:
                    return json.load(f)

            raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get run {run_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/reports")
    async def list_reports():
        """List available report files."""
        try:
            reports_dir = Path("outputs/reports")
            if not reports_dir.exists():
                return []

            reports = []
            for report_file in sorted(reports_dir.glob("*.json")):
                reports.append({
                    "name": report_file.stem,
                    "filename": report_file.name,
                    "size_bytes": report_file.stat().st_size
                })

            return reports
        except Exception as e:
            logger.error(f"Failed to list reports: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/reports/{report_name}")
    async def get_report(report_name: str):
        """Get a specific report file."""
        try:
            # Add .json extension if not provided
            if not report_name.endswith('.json'):
                report_name += '.json'

            report_file = Path(f"outputs/reports/{report_name}")

            if not report_file.exists():
                raise HTTPException(status_code=404, detail=f"Report {report_name} not found")

            # Security check: ensure file is within reports directory
            if not report_file.resolve().is_relative_to(Path("outputs/reports").resolve()):
                raise HTTPException(status_code=403, detail="Access denied")

            with open(report_file, 'r') as f:
                return json.load(f)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get report {report_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/logs/{run_id}")
    async def get_log(run_id: str):
        """Get log file for a specific run."""
        try:
            logs_dir = Path("outputs/logs")

            # Find log file matching the run_id
            log_files = list(logs_dir.glob(f"*{run_id}*.log"))

            if not log_files:
                raise HTTPException(status_code=404, detail=f"Log for run {run_id} not found")

            log_file = log_files[0]  # Use first matching log

            # Read log file
            with open(log_file, 'r') as f:
                content = f.read()

            return {
                "run_id": run_id,
                "filename": log_file.name,
                "content": content,
                "lines": len(content.splitlines())
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get log for {run_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/models")
    async def get_models_comparison():
        """Get model comparison data from the latest training report."""
        try:
            report_file = Path("outputs/reports/model_training_report.json")

            if not report_file.exists():
                raise HTTPException(status_code=404, detail="Model training report not found")

            with open(report_file, 'r') as f:
                data = json.load(f)

            # Extract relevant model comparison data
            return {
                "problem_type": data.get("problem_type"),
                "best_model": data.get("best_model"),
                "best_score": data.get("best_score"),
                "scoring_metric": data.get("scoring_metric"),
                "all_results": data.get("all_results", {}),
                "gpu_used": data.get("gpu_acceleration", {}).get("gpu_used_for_best_model", False)
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get models comparison: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # ========================================================================
    # WebSocket Endpoint
    # ========================================================================

    @app.websocket("/ws/progress")
    async def websocket_endpoint(websocket: WebSocket):
        """WebSocket endpoint for real-time progress updates."""
        await ws_manager.connect(websocket)
        try:
            # Keep connection alive and listen for client messages
            while True:
                # Wait for any message from client (ping/pong)
                await websocket.receive_text()
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            ws_manager.disconnect(websocket)

    # ========================================================================
    # Lifecycle Events
    # ========================================================================

    @app.on_event("startup")
    async def startup_event():
        """Start background tasks on server startup."""
        import asyncio
        logger.info("Starting FastAPI server...")
        # Get the current event loop and pass it to the watcher
        loop = asyncio.get_running_loop()
        ws_manager.start_watching(loop=loop)
        logger.info("FastAPI server started successfully")

    @app.on_event("shutdown")
    async def shutdown_event():
        """Cleanup on server shutdown."""
        logger.info("Shutting down FastAPI server...")
        ws_manager.stop_watching()
        await ws_manager.close_all()
        logger.info("FastAPI server shut down successfully")

    # ========================================================================
    # Static Files (for production React build)
    # ========================================================================

    # Check if React build exists
    dashboard_build = Path("dashboard/dist")
    if dashboard_build.exists():
        # Serve React static files
        app.mount("/assets", StaticFiles(directory=str(dashboard_build / "assets")), name="assets")

        @app.get("/")
        async def serve_react_app():
            """Serve React app index.html."""
            index_file = dashboard_build / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return {"message": "Dashboard not built. Run: cd dashboard && npm run build"}

    else:
        @app.get("/")
        async def root():
            """Root endpoint when React app is not built."""
            return {
                "message": "ML Pipeline Dashboard API",
                "version": "1.0.0",
                "docs": "/docs",
                "status": "React dashboard not built. For development, run React dev server separately."
            }

    return app
