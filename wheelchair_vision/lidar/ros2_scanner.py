"""
ROS 2 RPLiDAR A1M8 Subscriber, 360° Obstacle Clustering & Autonomous Path Planner.
"""

import os
import time
import math
import signal
import atexit
import threading
import subprocess
import numpy as np

from ..config import config


class ROS2LidarScanner:
    """
    Subscribes to the ROS 2 /scan topic (sensor_msgs/msg/LaserScan) in a background
    thread, automatically launches sllidar_node if not already running, and performs:
      1. 360-degree 6-sector safety evaluation
      2. Euclidean jump-distance obstacle clustering (distance, angle, width in meters)
      3. Vector Field / Passable-Gap autonomous path finding for the wheelchair
    """

    SECTOR_DEFS = {
        "front_left": (-45.0, -15.0),
        "front_center": (-15.0, 15.0),
        "front_right": (15.0, 45.0),
        "left_flank": (-135.0, -45.0),
        "right_flank": (45.0, 135.0),
        # "rear" is handled as |angle| > 135.0
    }

    def __init__(self, shared_state, control_settings):
        self.shared_state = shared_state
        self.control_settings = control_settings
        self.lock = threading.Lock()

        # Latest scan arrays (azimuth_deg in [-180, 180], range_m)
        # Convention: 0° = Wheelchair Front, +90° = Right, -90° = Left, ±180° = Rear
        self.angles_deg = np.array([], dtype=np.float32)
        self.ranges_m = np.array([], dtype=np.float32)
        self.last_scan_time = 0.0
        self.scan_hz = 0.0
        self.connected = False
        self.status_msg = "INITIALIZING ROS 2..."
        self.point_count = 0

        # 6-Sector distances and states
        self.sectors = {
            name: {"min_dist": 9.9, "blocked": False, "closest_angle": 0.0}
            for name in ["front_left", "front_center", "front_right", "left_flank", "right_flank", "rear"]
        }
        self.overall_min_dist = 9.9
        self.overall_closest_angle = 0.0

        # Clustered discrete obstacles & autonomous path planner state
        self.obstacles = []
        self.best_heading_deg = 0.0
        self.best_path_clearance_m = 3.5

        self._sllidar_proc = None
        self._running = True
        atexit.register(self.shutdown)

        self._thread = threading.Thread(target=self._ros2_worker, daemon=True)
        self._thread.start()

    def _maybe_start_sllidar_driver(self):
        """
        If /dev/ttyUSB0 exists and no /scan publisher is active yet,
        automatically launch sllidar_node in the background.
        """
        if self._sllidar_proc is not None:
            return
        if not os.path.exists(config.LIDAR_SERIAL_PORT):
            self.status_msg = f"WAITING FOR {config.LIDAR_SERIAL_PORT}..."
            return

        try:
            check = subprocess.run(
                ["pgrep", "-f", "sllidar_node"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=1.0,
            )
            if check.returncode == 0:
                return
        except Exception:
            pass

        sllidar_bin = os.path.expanduser(
            "~/lidar_ws/install/sllidar_ros2/lib/sllidar_ros2/sllidar_node"
        )
        if not os.path.isfile(sllidar_bin):
            self.status_msg = "sllidar_node binary not found in ~/lidar_ws"
            return

        try:
            self.status_msg = "STARTING RPLiDAR A1 DRIVER..."
            cmd = (
                f"source /opt/ros/jazzy/setup.bash && "
                f"source ~/lidar_ws/install/setup.bash && "
                f"exec {sllidar_bin} --ros-args "
                f"-p channel_type:=serial "
                f"-p serial_port:={config.LIDAR_SERIAL_PORT} "
                f"-p serial_baudrate:=115200 "
                f"-p frame_id:=laser "
                f"-p inverted:=false "
                f"-p angle_compensate:=true "
                f"-p scan_mode:=Sensitivity"
            )
            self._sllidar_proc = subprocess.Popen(
                ["bash", "-c", cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid,
            )
        except Exception as exc:
            self.status_msg = f"Driver launch error: {exc}"

    def _ros2_worker(self):
        try:
            import rclpy
            from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
            from sensor_msgs.msg import LaserScan
        except ImportError as exc:
            self.status_msg = f"ROS 2 rclpy not loaded ({exc})"
            return

        try:
            if not rclpy.ok():
                rclpy.init()
        except Exception:
            pass

        node = rclpy.create_node("wheelchair_lidar_perception")
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        node.create_subscription(
            LaserScan,
            config.LIDAR_TOPIC,
            self._on_scan_msg,
            qos,
        )

        self.status_msg = "WAITING FOR /scan..."
        start_wait = time.time()

        while self._running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.time()

            if self.last_scan_time > 0 and (now - self.last_scan_time) > 1.8:
                with self.lock:
                    self.connected = False
                    self.status_msg = "LIDAR STREAM TIMEOUT - RECONNECTING..."
                with self.shared_state.lock:
                    self.shared_state.lidar_connected = False

            if not self.connected and (now - start_wait) > 1.5:
                self._maybe_start_sllidar_driver()
                start_wait = now

    @staticmethod
    def _cluster_obstacles(azimuth_deg: np.ndarray, ranges_m: np.ndarray, max_cluster_range: float = 3.5):
        """
        Segments 360° laser points into discrete physical obstacles using adaptive
        Euclidean jump-distance clustering. Returns the closest obstacles with their
        exact distance (m), angle (deg), width (m), and Cartesian bounding box.
        """
        mask = ranges_m <= max_cluster_range
        if not np.any(mask):
            return []

        ang = azimuth_deg[mask]
        rng = ranges_m[mask]

        # Sort points by azimuth angle [-180°, +180°]
        order = np.argsort(ang)
        ang = ang[order]
        rng = rng[order]

        rad = np.radians(ang)
        xs = rng * np.sin(rad)
        ys = rng * np.cos(rad)

        # Jump-distance between consecutive rays
        dx = np.diff(xs)
        dy = np.diff(ys)
        step_dist = np.hypot(dx, dy)
        # Adaptive threshold: 14cm base or 10% of range, max 60cm cluster span split
        thresh = np.maximum(0.14, 0.10 * rng[:-1])
        split_indices = np.where(step_dist > thresh)[0] + 1

        clusters_idx = np.split(np.arange(len(rng)), split_indices)
        obstacles = []

        for c_idx in clusters_idx:
            if len(c_idx) < 4:
                continue  # Ignore isolated 1-3 ray noise spikes

            c_rng = rng[c_idx]
            c_ang = ang[c_idx]
            c_x = xs[c_idx]
            c_y = ys[c_idx]

            min_i = int(np.argmin(c_rng))
            min_dist = float(c_rng[min_i])
            closest_ang = float(c_ang[min_i])

            # Physical span / width of cluster
            width_m = float(math.hypot(c_x[-1] - c_x[0], c_y[-1] - c_y[0]))
            x1_m, x2_m = float(np.min(c_x)), float(np.max(c_x))
            y1_m, y2_m = float(np.min(c_y)), float(np.max(c_y))

            obstacles.append(
                {
                    "min_dist_m": round(min_dist, 2),
                    "angle_deg": round(closest_ang, 1),
                    "width_m": round(width_m, 2),
                    "cx_m": round(float(np.mean(c_x)), 2),
                    "cy_m": round(float(np.mean(c_y)), 2),
                    "x1_m": x1_m,
                    "y1_m": y1_m,
                    "x2_m": x2_m,
                    "y2_m": y2_m,
                    "rays": len(c_idx),
                }
            )

        # Sort by closest distance to the wheelchair and assign IDs #1..#8
        obstacles.sort(key=lambda o: o["min_dist_m"])
        top_obstacles = obstacles[:8]
        for idx, obs in enumerate(top_obstacles, start=1):
            obs["id"] = idx
        return top_obstacles

    @staticmethod
    def _compute_best_path(azimuth_deg: np.ndarray, ranges_m: np.ndarray, danger_dist_m: float):
        """
        Vector Field Histogram (VFH) / Passable Corridor Path Planner:
        Evaluates candidate steering headings across [-75°, +75°] accounting for
        the physical wheelchair width (config.WHEELCHAIR_WIDTH_M).
        Returns (best_heading_deg, clearance_m).
        """
        candidate_headings = np.arange(-75.0, 76.0, 5.0)
        best_heading = 0.0
        best_score = -1e9
        best_clearance = 0.0

        half_width = config.WHEELCHAIR_WIDTH_M * 0.52  # ~0.35m lateral clearance each side

        # Consider frontal hemisphere rays [-105°, +105°]
        front_mask = np.abs(azimuth_deg) <= 105.0
        f_ang = azimuth_deg[front_mask]
        f_rng = ranges_m[front_mask]

        if len(f_rng) == 0:
            return 0.0, 3.5

        for cand_deg in candidate_headings:
            # Angular offset of rays relative to this candidate heading
            rel_deg = f_ang - cand_deg
            in_cone = np.abs(rel_deg) <= 42.0
            if not np.any(in_cone):
                clearance = 3.5
            else:
                c_rng = f_rng[in_cone]
                c_rel_rad = np.radians(rel_deg[in_cone])
                # Lateral distance of each point from the candidate path centerline
                lat_dist = np.abs(c_rng * np.sin(c_rel_rad))
                forward_dist = c_rng * np.cos(c_rel_rad)

                # Points that physically intersect the wheelchair's corridor width
                in_corridor = (lat_dist <= half_width) & (forward_dist > 0.0)
                if np.any(in_corridor):
                    corr_dists = c_rng[in_corridor]
                    clearance = float(np.percentile(corr_dists, 5)) if len(corr_dists) >= 4 else float(np.min(corr_dists))
                else:
                    clearance = 3.5

            clearance_capped = min(3.5, clearance)
            # Penalize paths blocked within danger_dist_m heavily; slightly prefer straight ahead (0°)
            safety_bonus = 2.0 if clearance >= danger_dist_m else -3.0
            heading_penalty = 0.012 * abs(cand_deg)
            score = clearance_capped + safety_bonus - heading_penalty

            if score > best_score:
                best_score = score
                best_heading = float(cand_deg)
                best_clearance = round(clearance_capped, 2)

        return best_heading, best_clearance

    def _on_scan_msg(self, msg):
        now = time.time()
        if self.last_scan_time > 0:
            dt = now - self.last_scan_time
            if 0.01 < dt < 1.0:
                self.scan_hz = 0.8 * self.scan_hz + 0.2 * (1.0 / dt)
        else:
            self.scan_hz = 10.0
        self.last_scan_time = now

        ranges = np.asarray(msg.ranges, dtype=np.float32)
        n_pts = len(ranges)
        if n_pts == 0:
            return

        raw_angles_rad = msg.angle_min + np.arange(n_pts, dtype=np.float32) * msg.angle_increment
        raw_angles_deg = np.degrees(raw_angles_rad)

        valid_mask = (
            np.isfinite(ranges)
            & (ranges >= config.LIDAR_MIN_RANGE_M)
            & (ranges <= config.LIDAR_MAX_RANGE_M)
        )
        valid_ranges = ranges[valid_mask]
        valid_ros_deg = raw_angles_deg[valid_mask]

        with self.shared_state.lock:
            angle_offset = self.control_settings.get("lidar_angle_offset_deg", config.LIDAR_ANGLE_OFFSET_DEG)
            danger_dist_m = self.control_settings.get("lidar_danger_m", config.LIDAR_DEFAULT_DANGER_M)

        azimuth_deg = (-valid_ros_deg + 180.0 + angle_offset + 180.0) % 360.0 - 180.0

        # 1. Evaluate 6 safety sectors
        new_sectors = {}
        for sec_name, (a_min, a_max) in self.SECTOR_DEFS.items():
            mask = (azimuth_deg >= a_min) & (azimuth_deg <= a_max)
            sec_r = valid_ranges[mask]
            sec_a = azimuth_deg[mask]
            if len(sec_r) > 0:
                idx = int(np.argmin(sec_r))
                min_d = float(sec_r[idx])
                c_ang = float(sec_a[idx])
                close_count = int(np.sum(sec_r <= danger_dist_m))
                is_blk = (min_d <= danger_dist_m) and (close_count >= 2)
            else:
                min_d = 9.9
                c_ang = (a_min + a_max) * 0.5
                is_blk = False
            new_sectors[sec_name] = {
                "min_dist": round(min_d, 2),
                "blocked": is_blk,
                "closest_angle": round(c_ang, 1),
            }

        rear_mask = np.abs(azimuth_deg) > 135.0
        rear_r = valid_ranges[rear_mask]
        rear_a = azimuth_deg[rear_mask]
        if len(rear_r) > 0:
            idx = int(np.argmin(rear_r))
            min_d = float(rear_r[idx])
            c_ang = float(rear_a[idx])
            close_count = int(np.sum(rear_r <= danger_dist_m))
            is_blk = (min_d <= danger_dist_m) and (close_count >= 2)
        else:
            min_d = 9.9
            c_ang = 180.0
            is_blk = False
        new_sectors["rear"] = {
            "min_dist": round(min_d, 2),
            "blocked": is_blk,
            "closest_angle": round(c_ang, 1),
        }

        if len(valid_ranges) > 0:
            g_idx = int(np.argmin(valid_ranges))
            overall_min = round(float(valid_ranges[g_idx]), 2)
            overall_ang = round(float(azimuth_deg[g_idx]), 1)
        else:
            overall_min = 9.9
            overall_ang = 0.0

        # 2. Cluster 360° points into discrete Obstacles (#1, #2, ...)
        obstacles = self._cluster_obstacles(azimuth_deg, valid_ranges)

        # 3. Compute Autonomous Best Free-Space Path Heading
        best_heading_deg, best_clearance_m = self._compute_best_path(
            azimuth_deg, valid_ranges, danger_dist_m
        )

        with self.lock:
            self.angles_deg = azimuth_deg
            self.ranges_m = valid_ranges
            self.sectors = new_sectors
            self.overall_min_dist = overall_min
            self.overall_closest_angle = overall_ang
            self.obstacles = obstacles
            self.best_heading_deg = best_heading_deg
            self.best_path_clearance_m = best_clearance_m
            self.point_count = len(valid_ranges)
            self.connected = True
            self.status_msg = "ACTIVE (RPLiDAR A1M8)"

        with self.shared_state.lock:
            self.shared_state.lidar_connected = True
            self.shared_state.lidar_hz = round(self.scan_hz, 1)
            self.shared_state.lidar_sectors = new_sectors
            self.shared_state.lidar_closest_dist = overall_min
            self.shared_state.lidar_closest_angle = overall_ang
            self.shared_state.lidar_points_count = len(valid_ranges)
            self.shared_state.lidar_obstacles = obstacles
            self.shared_state.lidar_best_heading_deg = best_heading_deg
            self.shared_state.lidar_best_clearance_m = best_clearance_m

    def get_snapshot(self):
        """Returns a thread-safe copy of the current LiDAR scan, obstacles, and path state."""
        with self.lock:
            return {
                "connected": self.connected,
                "status_msg": self.status_msg,
                "scan_hz": round(self.scan_hz, 1),
                "point_count": self.point_count,
                "angles_deg": self.angles_deg.copy(),
                "ranges_m": self.ranges_m.copy(),
                "sectors": dict(self.sectors),
                "overall_min_dist": self.overall_min_dist,
                "overall_closest_angle": self.overall_closest_angle,
                "obstacles": list(self.obstacles),
                "best_heading_deg": self.best_heading_deg,
                "best_path_clearance_m": self.best_path_clearance_m,
            }

    def get_min_distance_in_angle_window(self, deg_min: float, deg_max: float):
        """
        Queries the minimum LiDAR distance (in meters) within the specified
        frontal azimuth window [deg_min, deg_max]. Used for Camera + LiDAR Sensor Fusion.
        """
        with self.lock:
            if not self.connected or len(self.ranges_m) == 0:
                return None
            mask = (self.angles_deg >= deg_min) & (self.angles_deg <= deg_max)
            sub_ranges = self.ranges_m[mask]
            if len(sub_ranges) == 0:
                return None
            return float(np.percentile(sub_ranges, 10)) if len(sub_ranges) >= 3 else float(np.min(sub_ranges))

    def shutdown(self):
        self._running = False
        if self._sllidar_proc is not None:
            try:
                os.killpg(os.getpgid(self._sllidar_proc.pid), signal.SIGTERM)
            except Exception:
                pass
            self._sllidar_proc = None
