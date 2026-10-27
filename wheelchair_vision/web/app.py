"""
Flask Application Factory for the Smart Wheelchair Web Dashboard.
"""

import os
from flask import Flask, Response, render_template, jsonify
from ..config import config


def create_app(frame_streamer, lidar_streamer, shared_state, control_settings, telemetry):
    template_dir = os.path.join(os.path.dirname(__file__), "templates")
    app = Flask(__name__, template_folder=template_dir)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video_feed")
    def video_feed():
        return Response(
            frame_streamer.generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/lidar_feed")
    def lidar_feed():
        return Response(
            lidar_streamer.generate(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.route("/telemetry")
    def get_telemetry():
        with shared_state.lock:
            return jsonify(telemetry)

    @app.route("/control/<action>", methods=["POST"])
    def control(action):
        with shared_state.lock:
            if action == "line_up":
                control_settings["danger_zone_y_ratio"] = max(
                    0.28, control_settings["danger_zone_y_ratio"] - 0.04
                )
            elif action == "line_down":
                control_settings["danger_zone_y_ratio"] = min(
                    0.85, control_settings["danger_zone_y_ratio"] + 0.04
                )
            elif action == "toggle_unknown":
                control_settings["use_unknown_edge_helper"] = not control_settings[
                    "use_unknown_edge_helper"
                ]
            elif action == "toggle_edges":
                control_settings["show_edge_overlay"] = not control_settings[
                    "show_edge_overlay"
                ]
            elif action == "lidar_danger_up":
                control_settings["lidar_danger_m"] = round(
                    min(2.50, control_settings["lidar_danger_m"] + 0.10), 2
                )
            elif action == "lidar_danger_down":
                control_settings["lidar_danger_m"] = round(
                    max(0.30, control_settings["lidar_danger_m"] - 0.10), 2
                )
            elif action == "lidar_rotate":
                control_settings["lidar_angle_offset_deg"] = (
                    control_settings["lidar_angle_offset_deg"] + 90.0
                ) % 360.0
            elif action == "lidar_align_left":
                control_settings["lidar_angle_offset_deg"] = (
                    control_settings["lidar_angle_offset_deg"] - 5.0
                ) % 360.0
            elif action == "lidar_align_right":
                control_settings["lidar_angle_offset_deg"] = (
                    control_settings["lidar_angle_offset_deg"] + 5.0
                ) % 360.0
            elif action == "lidar_zoom":
                zoom_levels = [2.5, 3.5, 6.0]
                cur = control_settings["lidar_radar_max_m"]
                idx = zoom_levels.index(cur) if cur in zoom_levels else 1
                control_settings["lidar_radar_max_m"] = zoom_levels[(idx + 1) % len(zoom_levels)]
            elif action == "toggle_clusters":
                control_settings["show_lidar_clusters"] = not control_settings.get(
                    "show_lidar_clusters", True
                )
            elif action == "toggle_lidar_cam":
                control_settings["show_lidar_on_cam"] = not control_settings.get(
                    "show_lidar_on_cam", True
                )
            elif action == "toggle_estop":
                control_settings["estop_active"] = not control_settings.get(
                    "estop_active", False
                )
        return jsonify({"ok": True})

    return app
