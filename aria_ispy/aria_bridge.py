import argparse
import sys
import time
from pathlib import Path

import aria.sdk as aria
import cv2
import numpy as np
import torch

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from common import ctrl_c_handler, quit_keypress, update_iptables
from inference import infer
from projectaria_tools.core.calibration import device_calibration_from_json_string
from projectaria_tools.core.mps import EyeGaze
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import ImageDataRecord
from state_io import write_player_state
from vision_tv import (
    detect_apriltags,
    estimate_rgb_to_tv_homography,
    load_layout,
    make_apriltag_detector,
    map_point_homography,
    tv_to_ispy,
)


def resolve_app_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return APP_DIR / path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--player", required=True, choices=["p1", "p2"])
    p.add_argument("--interface", default="usb", choices=["usb", "wifi"])
    p.add_argument("--device-ip", default=None)
    p.add_argument("--profile", default="profile18")
    p.add_argument("--device", default="mps")
    p.add_argument("--update_iptables", action="store_true")
    p.add_argument("--layout", default="runtime/tv_layout.json")
    p.add_argument("--state-dir", default="runtime/state")
    p.add_argument("--gaze-depth-m", type=float, default=2.0)
    p.add_argument("--tag-interval", type=int, default=15,
                    help="Re-detect AprilTags every N frames (default 15)")
    p.add_argument("--write-fps", type=float, default=15.0,
                    help="Max state writes per second to disk (default 15)")
    return p.parse_args()


class GazeEMA:
    """Exponential moving average filter for 2-D gaze coordinates."""

    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self._x = None
        self._y = None

    def update(self, xy):
        if xy is None:
            return None
        x, y = xy
        if self._x is None:
            self._x, self._y = x, y
        else:
            self._x += self.alpha * (x - self._x)
            self._y += self.alpha * (y - self._y)
        return [self._x, self._y]


class StreamingClientObserver:
    """
    Aria-only frame source.
    No webcam input is allowed anywhere in this file.
    """

    def __init__(self):
        self.rgb_image = None
        self.eye_image = None
        self.last_rgb_time_ns = None
        self.last_eye_time_ns = None

    def on_image_received(self, image: np.ndarray, record: ImageDataRecord):
        if record.camera_id == aria.CameraId.Rgb:
            self.rgb_image = image
            self.last_rgb_time_ns = time.time_ns()

        elif record.camera_id == aria.CameraId.EyeTrack:
            self.eye_image = image
            self.last_eye_time_ns = time.time_ns()


def main():
    args = parse_args()
    layout_path = resolve_app_path(args.layout)
    state_dir = resolve_app_path(args.state_dir)

    if args.update_iptables:
        update_iptables()

    if not layout_path.exists():
        raise SystemExit(
            f"Missing {layout_path}. Generate it first:\n"
            "  cd /Users/patrickpuma/Github/aria_realtime_gaze/aria_ispy\n"
            "  python make_tv_scene.py --image assets/ispy.jpg\n"
            "Then display runtime/tv_scene.png fullscreen on the TV."
        )

    layout = load_layout(layout_path)
    tag_detector = make_apriltag_detector()

    aria.set_log_level(aria.Level.Info)

    # 1. Connect to the Aria glasses.
    device_client = aria.DeviceClient()
    client_config = aria.DeviceClientConfig()

    if args.device_ip:
        client_config.ip_v4_address = args.device_ip

    device_client.set_client_config(client_config)
    device = device_client.connect()

    # 2. Configure streaming.
    streaming_manager = device.streaming_manager
    streaming_client = streaming_manager.streaming_client

    streaming_config = aria.StreamingConfig()
    streaming_config.profile_name = args.profile

    if args.interface == "usb":
        streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True

    streaming_manager.streaming_config = streaming_config

    # 3. Retrieve Aria factory calibration. This replaces checkerboard calibration.
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")

    # 4. Start streaming and subscribe to RGB + EyeTrack.
    streaming_manager.start_streaming()

    config = streaming_client.subscription_config
    config.subscriber_data_type = (
        aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    )
    streaming_client.subscription_config = config

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 5. Load eye-gaze inference model from the repository root.
    inference_model = infer.EyeGazeInference(
        str(REPO_DIR / "model" / "weights.pth"),
        str(REPO_DIR / "model" / "config.yaml"),
        args.device,
    )

    gaze_tv_filter = GazeEMA(alpha=0.3)

    TAG_REDETECT_INTERVAL = args.tag_interval
    WRITE_MIN_INTERVAL_S = 1.0 / args.write_fps

    print(f"[{args.player}] Aria bridge started")
    print(f"[{args.player}] Interface: {args.interface}")
    print(f"[{args.player}] Writing state to {state_dir}")

    frame_count = 0
    last_write_time = 0.0
    cached_H = None
    cached_h_debug = None
    cached_detections = {}

    try:
        with ctrl_c_handler() as ctrl_c:
            while not (quit_keypress() or ctrl_c):
                if observer.rgb_image is None or observer.eye_image is None:
                    time.sleep(0.005)
                    continue

                rgb_bgr = cv2.cvtColor(observer.rgb_image, cv2.COLOR_RGB2BGR)

                # 6. Eye-gaze inference from Aria EyeTrack image.
                eye_img = torch.tensor(observer.eye_image, device=args.device)
                preds, _lower, _upper = inference_model.predict(eye_img)
                preds = preds.detach().cpu().numpy()

                eye_gaze = EyeGaze
                eye_gaze.yaw = float(preds[0][0])
                eye_gaze.pitch = float(preds[0][1])

                # 7. Project gaze into the Aria RGB camera image.
                gaze_rgb = get_gaze_vector_reprojection(
                    eye_gaze,
                    "camera-rgb",
                    sensors_calib,
                    rgb_calib,
                    args.gaze_depth_m,
                )

                if gaze_rgb is None:
                    continue

                gaze_rgb_xy = [float(gaze_rgb[0]), float(gaze_rgb[1])]

                # 8. Detect AprilTags periodically; reuse cached homography otherwise.
                if frame_count % TAG_REDETECT_INTERVAL == 0 or cached_H is None:
                    detections = detect_apriltags(rgb_bgr, tag_detector)
                    H_rgb_to_tv, h_debug = estimate_rgb_to_tv_homography(
                        detections=detections,
                        layout=layout,
                    )
                    if H_rgb_to_tv is not None:
                        cached_H = H_rgb_to_tv
                        cached_h_debug = h_debug
                        cached_detections = detections
                else:
                    H_rgb_to_tv = cached_H
                    h_debug = cached_h_debug
                    detections = cached_detections

                frame_count += 1

                gaze_tv_xy = None
                gaze_ispy_xy = None

                if H_rgb_to_tv is not None:
                    gaze_tv_raw = map_point_homography(H_rgb_to_tv, gaze_rgb_xy)
                    gaze_tv_xy = gaze_tv_filter.update(gaze_tv_raw)
                    gaze_ispy_xy = tv_to_ispy(gaze_tv_xy, layout)

                # 9. Throttle disk writes + debug overlay to ~15 Hz.
                now = time.monotonic()
                if now - last_write_time < WRITE_MIN_INTERVAL_S:
                    continue
                last_write_time = now

                debug_frame = rgb_bgr.copy()

                gx, gy = int(gaze_rgb_xy[0]), int(gaze_rgb_xy[1])
                cv2.circle(debug_frame, (gx, gy), 16, (0, 255, 0), -1)

                for tag_id, corners in detections.items():
                    pts = corners.astype(np.int32)
                    cv2.polylines(debug_frame, [pts], True, (255, 0, 0), 3)
                    x, y = pts[0]
                    cv2.putText(
                        debug_frame,
                        str(tag_id),
                        (int(x), int(y) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 0, 0),
                        2,
                    )

                debug_frame_display = cv2.rotate(debug_frame, cv2.ROTATE_90_CLOCKWISE)

                # 10. Write state for dashboard.
                state = {
                    "player_id": args.player,
                    "timestamp_unix_ms": int(time.time() * 1000),
                    "gaze_rgb": gaze_rgb_xy,
                    "gaze_tv": gaze_tv_xy,
                    "gaze_ispy": gaze_ispy_xy,
                    "tags_found": sorted(list(detections.keys())),
                    "homography_ok": H_rgb_to_tv is not None,
                    "homography_debug": h_debug,
                    "H_rgb_to_tv": H_rgb_to_tv.tolist()
                    if H_rgb_to_tv is not None
                    else None,
                    "source": "aria",
                    "forbidden_webcam_used": False,
                    "debug_frame_rotation": "clockwise_90",
                }

                write_player_state(
                    state_dir=state_dir,
                    player_id=args.player,
                    frame_bgr=debug_frame_display,
                    state=state,
                )
    finally:
        streaming_client.unsubscribe()
        streaming_manager.stop_streaming()
        device_client.disconnect(device)


if __name__ == "__main__":
    main()
