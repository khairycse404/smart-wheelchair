"""
Web interface subpackage for live telemetry dashboard and MJPEG video streaming.
"""

from .app import create_app

__all__ = ["create_app"]
