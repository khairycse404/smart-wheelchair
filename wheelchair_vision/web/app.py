"""
Flask Application Factory for the Smart Wheelchair Control & Visualization GUI.
"""

import os
from flask import Flask, Response, render_template, jsonify, request
from ..config import config


def create_app(
    frame_streamer,
    lidar_streamer,
    arduino_bridge,
    shared_state,
    control_settings,
    telemetry,
):
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

    @app.route("/api/settings", methods=["GET"])
    def get_settings():
        with shared_state.lock:
            return jsonify(control_settings)

    @app.route("/api/mode", methods=["POST"])
    def set_drive_mode():
        data = request.get_json(silent=True) or {}
        mode = str(data.get("mode", "STOP")).upper().strip()
        if mode not in ("STOP", "MANUAL", "TEST", "AUTO"):
            mode = "STOP"
        with shared_state.lock:
            control_settings["drive_mode"] = mode
            control_settings["manual_action"] = "STOP"
            if mode == "STOP":
                telemetry["speed_pct"] = 0
                telemetry["steer_deg"] = 0
        if mode != "TEST":
            arduino_bridge.trigger_motor_test("NONE") if False else None
        return jsonify({"ok": True, "drive_mode": mode})

    @app.route("/api/manual", methods=["POST"])
    def manual_drive():
        data = request.get_json(silent=True) or {}
        action = str(data.get("action", "STOP")).upper().strip()
        speed_pct = data.get("speed_pct")
        with shared_state.lock:
            if speed_pct is not None:
                try:
                    control_settings["manual_speed_pct"] = max(
                        10, min(100, int(speed_pct))
                    )
                except ValueError:
                    pass
            if action in ("FWD", "BACK", "LEFT", "RIGHT", "FWD_LEFT", "FWD_RIGHT", "STOP"):
                control_settings["manual_action"] = action
                if action != "STOP":
                    control_settings["drive_mode"] = "MANUAL"
        return jsonify(
            {
                "ok": True,
                "drive_mode": control_settings["drive_mode"],
                "manual_action": control_settings["manual_action"],
                "manual_speed_pct": control_settings["manual_speed_pct"],
            }
        )

    @app.route("/api/test_motor", methods=["POST"])
    def test_motor():
        data = request.get_json(silent=True) or {}
        action = str(data.get("action", "NONE")).upper().strip()
        arduino_bridge.trigger_motor_test(action)
        return jsonify({"ok": True, "test_action": action, "drive_mode": "TEST"})

    @app.route("/api/arduino_config", methods=["POST"])
    def update_arduino_config():
        data = request.get_json(silent=True) or {}
        port_changed = False
        new_port = None

        with shared_state.lock:
            if "arduino_port" in data:
                req_port = str(data["arduino_port"]).strip() or "AUTO"
                if req_port != control_settings.get("arduino_port"):
                    port_changed = True
                    new_port = req_port
                control_settings["arduino_port"] = req_port

            for pin_key in (
                "pin_speed",
                "pin_steer",
                "pin_relay",
                "pin_estop",
                "pin_gps_rx",
                "pin_gps_tx",
            ):
                if pin_key in data:
                    try:
                        val = max(0, min(53, int(data[pin_key])))
                        control_settings[pin_key] = val
                    except ValueError:
                        pass

            if "reverse_drive" in data:
                control_settings["reverse_drive"] = bool(data["reverse_drive"])
            if "reverse_steer" in data:
                control_settings["reverse_steer"] = bool(data["reverse_steer"])

        if port_changed:
            arduino_bridge.reconnect(new_port)
        arduino_bridge.sync_hardware_config()

        with shared_state.lock:
            return jsonify({"ok": True, "settings": dict(control_settings)})

    @app.route("/api/gps_toggle", methods=["POST"])
    def toggle_gps():
        data = request.get_json(silent=True) or {}
        with shared_state.lock:
            if "enabled" in data:
                control_settings["gps_enabled"] = bool(data["enabled"])
            else:
                control_settings["gps_enabled"] = not control_settings.get(
                    "gps_enabled", False
                )
            en = control_settings["gps_enabled"]

        arduino_bridge.sync_gps_state()
        return jsonify({"ok": True, "gps_enabled": en})

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
                control_settings["lidar_radar_max_m"] = zoom_levels[
                    (idx + 1) % len(zoom_levels)
                ]
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
                if control_settings["estop_active"]:
                    control_settings["manual_action"] = "STOP"
        return jsonify({"ok": True})

    return app
