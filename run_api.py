"""Run the FastAPI backend for the ML Pipeline Dashboard.

Usage:
    python run_api.py [--host HOST] [--port PORT] [--reload]

Serves the REST API and WebSocket stream at http://localhost:8000 by default.
If dashboard/dist exists (after `npm run build`), the built UI is served too.
"""

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the genML dashboard API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    # Server paths (outputs/, dashboard/dist) are relative to the project root.
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

    import uvicorn

    uvicorn.run(
        "src.genML.api:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
