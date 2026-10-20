"""
Flask Application Factory for the Smart Wheelchair Web Dashboard.
"""

import os
from flask import Flask, Response, render_template, jsonify
from ..config import config


def create_app(frame_streamer, shared_state, control_settings, telemetry):
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
        return jsonify({"ok": True})

    return app
