"""
Modular Serial Bridge between the HP Mini PC and the Arduino Motor/DAC & GPS Controller.

Responsibilities:
  1. Auto-detects or connects to the Arduino USB Serial port at 115200 baud.
  2. Arbitrates Drive Mode (STOP, MANUAL, TEST, AUTO) and streams 20 Hz control frames.
  3. Sends runtime pin customizations and motor polarity reversal flags from the Web GUI.
  4. Toggles the Arduino GPS Module ON/OFF and provides a clean manual integration hook
     for future GPS coordinate/waypoint logic.
  5. Computes live 12-bit DAC values (0..4095) and joystick voltages (0.50V..4.50V,
     2.50V neutral) even when running in unplugged simulation mode.
"""

import glob
import threading
import time

try:
    import serial
except ImportError:
    serial = None

from ..config import config

DAC_NEUTRAL = 2048
DAC_MIN_SAFE = 410
DAC_MAX_SAFE = 3686
DAC_MAX_SWING = 1638


class ArduinoSerialBridge:
    def __init__(self, shared_state, control_settings, telemetry):
        self.shared_state = shared_state
        self.control_settings = control_settings
        self.telemetry = telemetry

        self._ser = None
        self._serial_lock = threading.Lock()
        self._pending_lines = []

        # Simulated test sequence state (mirrors Arduino firmware when unplugged or active)
        self._active_test_name = "NONE"
        self._test_start_time = 0.0
        self._last_cfg_sync_time = 0.0

        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # -------------------------------------------------------------------------
    # Port Discovery & Connection
    # -------------------------------------------------------------------------
    def _discover_port(self) -> str:
        requested = self.control_settings.get("arduino_port", config.ARDUINO_SERIAL_PORT)
        if requested and requested.upper() != "AUTO":
            return requested

        # Prefer /dev/ttyACM* (Arduino Uno / Mega native CDC)
        acm_ports = sorted(glob.glob("/dev/ttyACM*"))
        if acm_ports:
            return acm_ports[0]

        # Fall back to /dev/ttyUSB* excluding the RPLiDAR A1M8 port (/dev/ttyUSB0)
        usb_ports = sorted(glob.glob("/dev/ttyUSB*"))
        lidar_port = getattr(config, "LIDAR_SERIAL_PORT", "/dev/ttyUSB0")
        for p in usb_ports:
            if p != lidar_port:
                return p
        return ""

    def _connect_if_needed(self):
        if serial is None:
            return False

        with self._serial_lock:
            if self._ser is not None and self._ser.is_open:
                return True

            port = self._discover_port()
            if not port:
                return False

            baud = int(self.control_settings.get("arduino_baud", config.ARDUINO_BAUD_RATE))
            try:
                self._ser = serial.Serial(port=port, baudrate=baud, timeout=0.02, write_timeout=0.05)
                time.sleep(1.6)  # Allow Arduino bootloader reset to complete
                self._queue_hardware_config_locked()
                self._queue_gps_state_locked()
                return True
            except Exception:
                self._ser = None
                return False

    def reconnect(self, new_port: str = None):
        with self.shared_state.lock:
            if new_port is not None:
                self.control_settings["arduino_port"] = new_port.strip() or "AUTO"

        with self._serial_lock:
            if self._ser is not None:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None

    # -------------------------------------------------------------------------
    # Runtime Commands from Web GUI
    # -------------------------------------------------------------------------
    def _queue_hardware_config_locked(self):
        with self.shared_state.lock:
            p_spd = int(self.control_settings.get("pin_speed", config.DEFAULT_PIN_SPEED))
            p_str = int(self.control_settings.get("pin_steer", config.DEFAULT_PIN_STEER))
            p_rly = int(self.control_settings.get("pin_relay", config.DEFAULT_PIN_RELAY))
            p_est = int(self.control_settings.get("pin_estop", config.DEFAULT_PIN_ESTOP))
            p_grx = int(self.control_settings.get("pin_gps_rx", config.DEFAULT_PIN_GPS_RX))
            p_gtx = int(self.control_settings.get("pin_gps_tx", config.DEFAULT_PIN_GPS_TX))
            rev_d = 1 if self.control_settings.get("reverse_drive", False) else 0
            rev_s = 1 if self.control_settings.get("reverse_steer", False) else 0

        cmd = f"CFG:{p_spd},{p_str},{p_rly},{p_est},{p_grx},{p_gtx},{rev_d},{rev_s}\n"
        self._pending_lines.append(cmd)

    def _queue_gps_state_locked(self):
        with self.shared_state.lock:
            en = 1 if self.control_settings.get("gps_enabled", False) else 0
        self._pending_lines.append(f"GPS:{en}\n")

    def sync_hardware_config(self):
        with self._serial_lock:
            self._queue_hardware_config_locked()

    def sync_gps_state(self):
        with self._serial_lock:
            self._queue_gps_state_locked()

    def trigger_motor_test(self, action: str):
        action_clean = action.upper().strip()
        if action_clean not in ("FWD", "BACK", "LEFT", "RIGHT", "NEUTRAL", "SWEEP", "NONE"):
            action_clean = "NONE"

        with self.shared_state.lock:
            self.control_settings["drive_mode"] = "TEST"
            self.control_settings["manual_action"] = "STOP"
            self._active_test_name = action_clean
            self._test_start_time = time.time()

        with self._serial_lock:
            self._pending_lines.append(f"TEST:{action_clean}\n")

    # -------------------------------------------------------------------------
    # DAC Math & Pre-Autonomous Test Generator
    # -------------------------------------------------------------------------
    @staticmethod
    def compute_dac_outputs(speed_pct: int, steer_deg: int, rev_drive: bool, rev_steer: bool):
        eff_speed = -speed_pct if rev_drive else speed_pct
        eff_steer = -steer_deg if rev_steer else steer_deg

        speed_offset = int((eff_speed * DAC_MAX_SWING) / 100)
        steer_offset = int((eff_steer * DAC_MAX_SWING) / 75)

        dac_y = max(DAC_MIN_SAFE, min(DAC_MAX_SAFE, DAC_NEUTRAL + speed_offset))
        dac_x = max(DAC_MIN_SAFE, min(DAC_MAX_SAFE, DAC_NEUTRAL + steer_offset))

        volt_y = round((dac_y * 5.0) / 4095.0, 2)
        volt_x = round((dac_x * 5.0) / 4095.0, 2)
        return dac_y, dac_x, volt_y, volt_x

    def _evaluate_test_sequence(self):
        if self._active_test_name == "NONE":
            return 0, 0, "TEST READY (SELECT PULSE)"

        elapsed = time.time() - self._test_start_time
        test_spd = 30
        test_str = 35
        act = self._active_test_name

        if act == "FWD":
            if elapsed < 1.5:
                return test_spd, 0, f"TESTING FORWARD (+{test_spd}%)"
        elif act == "BACK":
            if elapsed < 1.5:
                return -test_spd, 0, f"TESTING REVERSE (-{test_spd}%)"
        elif act == "LEFT":
            if elapsed < 1.5:
                return 0, -test_str, f"TESTING LEFT TURN (-{test_str} deg)"
        elif act == "RIGHT":
            if elapsed < 1.5:
                return 0, test_str, f"TESTING RIGHT TURN (+{test_str} deg)"
        elif act == "NEUTRAL":
            if elapsed < 1.5:
                return 0, 0, "TESTING NEUTRAL (2.50V CENTER)"
        elif act == "SWEEP":
            if elapsed < 1.2:
                return test_spd, 0, "SWEEP 1/4: FORWARD PULSE"
            elif elapsed < 1.6:
                return 0, 0, "SWEEP PAUSE (NEUTRAL)"
            elif elapsed < 2.8:
                return -test_spd, 0, "SWEEP 2/4: REVERSE PULSE"
            elif elapsed < 3.2:
                return 0, 0, "SWEEP PAUSE (NEUTRAL)"
            elif elapsed < 4.4:
                return 0, -test_str, "SWEEP 3/4: LEFT STEER PULSE"
            elif elapsed < 4.8:
                return 0, 0, "SWEEP PAUSE (NEUTRAL)"
            elif elapsed < 6.0:
                return 0, test_str, "SWEEP 4/4: RIGHT STEER PULSE"

        self._active_test_name = "NONE"
        return 0, 0, "TEST COMPLETE (NEUTRAL 2.50V)"

    def _compute_manual_command(self, manual_action: str, manual_speed_pct: int):
        spd = max(10, min(100, int(manual_speed_pct)))
        act = (manual_action or "STOP").upper()
        if act == "FWD":
            return spd, 0, f"MANUAL FORWARD ({spd}%)"
        if act == "BACK":
            return -spd, 0, f"MANUAL REVERSE (-{spd}%)"
        if act == "LEFT":
            return 0, -45, "MANUAL TURN LEFT (-45 deg)"
        if act == "RIGHT":
            return 0, 45, "MANUAL TURN RIGHT (+45 deg)"
        if act == "FWD_LEFT":
            return spd, -30, f"MANUAL FORWARD-LEFT ({spd}%)"
        if act == "FWD_RIGHT":
            return spd, 30, f"MANUAL FORWARD-RIGHT ({spd}%)"
        return 0, 0, "MANUAL STANDBY (NEUTRAL)"

    # -------------------------------------------------------------------------
    # GPS Telemetry Parser & Manual Expansion Hook
    # -------------------------------------------------------------------------
    def _process_gps_telemetry(self, parsed_kv: dict):
        """
        ========================================================================
        [MANUAL GPS INTEGRATION PLACEHOLDER - PYTHON SIDE]
        When GPS is toggled ON in the GUI, raw fields from the Arduino arrive
        in `parsed_kv` (GPS_EN, GPS_ST, GPS_FIX, SATS, LAT, LON).
        Add custom coordinate logging, Google Maps link generation, or waypoint
        heading calculation in the block below.
        ========================================================================
        """
        gps_en = parsed_kv.get("GPS_EN") == "1"
        gps_fix = parsed_kv.get("GPS_FIX") == "1"
        gps_st = parsed_kv.get("GPS_ST", "SEARCHING_SATS" if gps_en else "DISABLED")
        try:
            sats = int(parsed_kv.get("SATS", "0"))
        except ValueError:
            sats = 0
        try:
            lat = float(parsed_kv.get("LAT", "0.0"))
            lon = float(parsed_kv.get("LON", "0.0"))
        except ValueError:
            lat, lon = 0.0, 0.0

        # --- Custom manual GPS fields can be populated here later ---
        custom_note = "Ready for manual coordinate / waypoint logic"

        with self.shared_state.lock:
            self.telemetry["gps"].update(
                {
                    "enabled": gps_en,
                    "status": gps_st.replace("_", " "),
                    "has_fix": gps_fix,
                    "satellites": sats,
                    "latitude": round(lat, 6) if gps_fix else None,
                    "longitude": round(lon, 6) if gps_fix else None,
                    "custom_slot": custom_note,
                }
            )

    def _parse_arduino_status_line(self, line: str):
        if not line.startswith("STAT:"):
            return
        body = line[5:].strip()
        kv = {}
        for part in body.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip()

        self._process_gps_telemetry(kv)

        with self.shared_state.lock:
            try:
                if "DAC_Y" in kv:
                    self.telemetry["arduino"]["dac_speed"] = int(kv["DAC_Y"])
                if "DAC_X" in kv:
                    self.telemetry["arduino"]["dac_steer"] = int(kv["DAC_X"])
                if "V_Y" in kv:
                    self.telemetry["arduino"]["voltage_speed"] = float(kv["V_Y"])
                if "V_X" in kv:
                    self.telemetry["arduino"]["voltage_steer"] = float(kv["V_X"])
                if "RLY" in kv:
                    self.telemetry["arduino"]["relay_engaged"] = kv["RLY"] == "1"
            except ValueError:
                pass

    # -------------------------------------------------------------------------
    # Main Background 20 Hz Serial Loop
    # -------------------------------------------------------------------------
    def _worker_loop(self):
        last_connect_attempt = 0.0

        while self._running:
            now = time.time()
            connected = False

            if self._ser is not None and self._ser.is_open:
                connected = True
            elif now - last_connect_attempt >= 2.0:
                last_connect_attempt = now
                connected = self._connect_if_needed()

            with self.shared_state.lock:
                drive_mode = self.control_settings.get("drive_mode", "STOP").upper()
                estop_active = self.control_settings.get("estop_active", False)
                manual_action = self.control_settings.get("manual_action", "STOP")
                manual_speed = self.control_settings.get(
                    "manual_speed_pct", config.DEFAULT_MANUAL_SPEED_PCT
                )
                rev_drive = self.control_settings.get("reverse_drive", False)
                rev_steer = self.control_settings.get("reverse_steer", False)
                gps_enabled = self.control_settings.get("gps_enabled", False)

                auto_speed = int(self.telemetry.get("auto_speed_pct", 0))
                auto_steer = int(self.telemetry.get("auto_steer_deg", 0))
                auto_cmd_text = self.telemetry.get("auto_command", "STANDBY")

            # Arbitrate final output Speed (%) and Steer (deg) based on active Drive Mode
            if estop_active:
                out_speed, out_steer = 0, 0
                active_status = "EMERGENCY STOP ENGAGED (0% / 2.50V)"
            elif drive_mode == "STOP":
                out_speed, out_steer = 0, 0
                active_status = "STANDBY / STOPPED (NEUTRAL 2.50V)"
            elif drive_mode == "MANUAL":
                out_speed, out_steer, active_status = self._compute_manual_command(
                    manual_action, manual_speed
                )
            elif drive_mode == "TEST":
                out_speed, out_steer, active_status = self._evaluate_test_sequence()
            elif drive_mode == "AUTO":
                out_speed, out_steer = auto_speed, auto_steer
                active_status = f"AUTONOMOUS: {auto_cmd_text}"
            else:
                out_speed, out_steer = 0, 0
                active_status = "STANDBY"

            dac_y, dac_x, volt_y, volt_x = self.compute_dac_outputs(
                out_speed, out_steer, rev_drive, rev_steer
            )
            relay_state = (not estop_active) and (drive_mode in ("MANUAL", "TEST", "AUTO"))

            # Transmit to physical Arduino if connected
            active_port_name = "DISCONNECTED (SIMULATED PREVIEW)"
            if connected:
                with self._serial_lock:
                    try:
                        active_port_name = self._ser.port
                        while self._pending_lines:
                            pkt = self._pending_lines.pop(0)
                            self._ser.write(pkt.encode("ascii", errors="ignore"))

                        estop_flag = 1 if estop_active else 0
                        cmd_line = f"CMD:{drive_mode},{out_speed},{out_steer},{estop_flag}\n"
                        self._ser.write(cmd_line.encode("ascii", errors="ignore"))

                        while self._ser.in_waiting > 0:
                            raw = self._ser.readline().decode("ascii", errors="ignore").strip()
                            if raw:
                                self._parse_arduino_status_line(raw)
                    except Exception:
                        try:
                            self._ser.close()
                        except Exception:
                            pass
                        self._ser = None
                        connected = False
                        active_port_name = "DISCONNECTED (SIMULATED PREVIEW)"

            with self.shared_state.lock:
                self.telemetry["drive_mode"] = drive_mode
                self.telemetry["active_status"] = active_status
                self.telemetry["steer_deg"] = out_steer
                self.telemetry["speed_pct"] = out_speed
                self.telemetry["arduino"].update(
                    {
                        "connected": connected,
                        "port": active_port_name,
                        "baud": self.control_settings.get("arduino_baud", config.ARDUINO_BAUD_RATE),
                        "dac_speed": dac_y,
                        "dac_steer": dac_x,
                        "voltage_speed": volt_y,
                        "voltage_steer": volt_x,
                        "relay_engaged": relay_state,
                        "test_active": self._active_test_name,
                        "pin_speed": self.control_settings.get("pin_speed", config.DEFAULT_PIN_SPEED),
                        "pin_steer": self.control_settings.get("pin_steer", config.DEFAULT_PIN_STEER),
                        "pin_relay": self.control_settings.get("pin_relay", config.DEFAULT_PIN_RELAY),
                        "pin_estop": self.control_settings.get("pin_estop", config.DEFAULT_PIN_ESTOP),
                        "pin_gps_rx": self.control_settings.get("pin_gps_rx", config.DEFAULT_PIN_GPS_RX),
                        "pin_gps_tx": self.control_settings.get("pin_gps_tx", config.DEFAULT_PIN_GPS_TX),
                        "reverse_drive": rev_drive,
                        "reverse_steer": rev_steer,
                    }
                )
                if not connected:
                    self.telemetry["gps"].update(
                        {
                            "enabled": gps_enabled,
                            "status": "ENABLED (WAITING FOR ARDUINO UART)"
                            if gps_enabled
                            else "DISABLED",
                            "rx_pin": self.control_settings.get(
                                "pin_gps_rx", config.DEFAULT_PIN_GPS_RX
                            ),
                            "tx_pin": self.control_settings.get(
                                "pin_gps_tx", config.DEFAULT_PIN_GPS_TX
                            ),
                        }
                    )

            time.sleep(0.05)  # 20 Hz update loop
