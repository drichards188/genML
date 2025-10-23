"""
API package for ML Pipeline Dashboard.

This package provides the FastAPI backend for the real-time monitoring dashboard.
"""

from .server import create_app
from .websocket_manager import WebSocketManager

__all__ = ['create_app', 'WebSocketManager']
