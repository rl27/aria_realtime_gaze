import argparse
import sys

import aria.sdk as aria

import cv2
import numpy as np

from common import ctrl_c_handler, quit_keypress, update_iptables

from projectaria_tools.core.calibration import (
    device_calibration_from_json_string,
    distort_by_calibration,
    get_linear_camera_calibration,
)
from projectaria_tools.core.sensor_data import ImageDataRecord

##########

import argparse
import csv
import os

import rerun as rr
import torch

try:
    from inference import infer  # Try local imports first
except ImportError:
    from projectaria_eyetracking.inference import infer

from projectaria_tools.core import data_provider
from projectaria_tools.core.mps import EyeGaze, get_eyegaze_point_at_depth
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection
from projectaria_tools.core.sensor_data import SensorDataType, TimeDomain
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.utils.rerun_helpers import AriaGlassesOutline

from tqdm import tqdm

########

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        default="usb",
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux",
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip",
        help="IP address to connect to the device over wifi (legacy/shared)",
    )
    parser.add_argument(
        "--device-ip-a",
        help="IP address for device A over wifi (overrides --device-ip)",
    )
    parser.add_argument(
        "--device-ip-b",
        help="IP address for device B over wifi (overrides --device-ip)",
    )
    parser.add_argument(
        "--device-serial-a",
        type=str,
        default='1WM093701V1275',
        help="Serial number for device A.",
    )
    parser.add_argument(
        "--device-serial-b",
        type=str,
        default='1WM103501F1325',
        help="Serial number for device B.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="device to run inference on",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    #  Optional: Set SDK's log level to Trace or Debug for more verbose logs. Defaults to Info
    aria.set_log_level(aria.Level.Info)

    class StreamingClientObserver:
        def __init__(self):
            self.rgb_image = None
            self.eye_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            if record.camera_id == aria.CameraId.Rgb:
                self.rgb_image = image
            if record.camera_id == aria.CameraId.EyeTrack:
                self.eye_image = image

    def get_connected_serial(device):
        for attr in ("device_serial", "serial_number", "serial", "device_serial_number"):
            if hasattr(device, attr):
                return getattr(device, attr)
        if hasattr(device, "device_info"):
            info = device.device_info
            for attr in ("device_serial", "serial_number", "serial", "device_serial_number"):
                if hasattr(info, attr):
                    return getattr(info, attr)
        return None

    def connect_device(device_serial: str, device_ip: str | None):
        device_client = aria.DeviceClient()
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        else:
            client_config.device_serial = device_serial
        device_client.set_client_config(client_config)

        device = device_client.connect()
        connected_serial = get_connected_serial(device)
        print(
            f"Requested serial {device_serial}; connected serial {connected_serial or 'unknown'}"
        )
        streaming_manager = device.streaming_manager
        streaming_client = streaming_manager.streaming_client

        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = args.profile_name
        if args.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        if args.streaming_interface == "wifi":
            streaming_config.streaming_interface = aria.StreamingInterface.Wifi
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config

        sensors_calib_json = streaming_manager.sensors_calibration()
        sensors_calib = device_calibration_from_json_string(sensors_calib_json)
        rgb_calib = sensors_calib.get_camera_calib("camera-rgb")

        streaming_manager.start_streaming()

        config = streaming_client.subscription_config
        config.subscriber_data_type = (
            aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
        )
        config.message_queue_size[aria.StreamingDataType.Rgb] = 1
        config.message_queue_size[aria.StreamingDataType.EyeTrack] = 1
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        streaming_client.subscription_config = config

        observer = StreamingClientObserver()
        streaming_client.set_streaming_client_observer(observer)
        streaming_client.subscribe()

        return {
            "device_client": device_client,
            "device": device,
            "streaming_manager": streaming_manager,
            "streaming_client": streaming_client,
            "observer": observer,
            "sensors_calib": sensors_calib,
            "rgb_calib": rgb_calib,
        }

    device_a_ip = args.device_ip_a or args.device_ip
    device_b_ip = args.device_ip_b or args.device_ip
    if args.streaming_interface == "wifi" and (not device_a_ip or not device_b_ip):
        raise ValueError(
            "Wi-Fi streaming requires --device-ip-a and --device-ip-b (or --device-ip as a shared fallback)."
        )

    device_a = connect_device(args.device_serial_a, device_a_ip)
    device_b = connect_device(args.device_serial_b, device_b_ip)
    serial_a = get_connected_serial(device_a["device"])
    serial_b = get_connected_serial(device_b["device"])
    if serial_a and serial_b and serial_a == serial_b:
        print(
            "Warning: both connections resolved to the same device serial."
        )

    # 9. Render the streaming data until we close the window
    rgb_window_a = "RGB images - A"
    eye_window_a = "Eye tracking - A"
    rgb_window_b = "RGB images - B"
    eye_window_b = "Eye tracking - B"

    cv2.namedWindow(rgb_window_a, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window_a, 512, 512)
    cv2.setWindowProperty(rgb_window_a, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window_a, 50, 50)

    cv2.namedWindow(eye_window_a, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window_a, 512, 512)
    cv2.setWindowProperty(eye_window_a, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window_a, 600, 50)

    cv2.namedWindow(rgb_window_b, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window_b, 512, 512)
    cv2.setWindowProperty(rgb_window_b, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window_b, 50, 600)

    cv2.namedWindow(eye_window_b, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window_b, 512, 512)
    cv2.setWindowProperty(eye_window_b, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window_b, 600, 600)

    # 10. Set up inference model
    inference_model_a = infer.EyeGazeInference(
        f"{os.path.dirname(__file__)}/model/weights.pth",
        f"{os.path.dirname(__file__)}/model/config.yaml",
        args.device,
    )
    inference_model_b = infer.EyeGazeInference(
        f"{os.path.dirname(__file__)}/model/weights.pth",
        f"{os.path.dirname(__file__)}/model/config.yaml",
        args.device,
    )
    depth_m = 1

    rgb_image_a = None
    rgb_image_b = None
    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if device_a["observer"].rgb_image is not None:
                rgb_image_a = cv2.cvtColor(device_a["observer"].rgb_image, cv2.COLOR_BGR2RGB)

                if device_a["observer"].eye_image is not None:
                    cv2.imshow(eye_window_a, device_a["observer"].eye_image)

                    # input size: 240x640
                    img = torch.tensor(device_a["observer"].eye_image, device=args.device)
                    preds, lower, upper = inference_model_a.predict(img)
                    preds = preds.detach().cpu().numpy()
                    lower = lower.detach().cpu().numpy()
                    upper = upper.detach().cpu().numpy()
                    value_mapping = {
                        "yaw": preds[0][0],
                        "pitch": preds[0][1],
                        "yaw_lower": lower[0][0],
                        "pitch_lower": lower[0][1],
                        "yaw_upper": upper[0][0],
                        "pitch_upper": upper[0][1],
                    }

                    eye_gaze = EyeGaze
                    eye_gaze.yaw = value_mapping["yaw"]
                    eye_gaze.pitch = value_mapping["pitch"]

                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze,
                        "camera-rgb",
                        device_a["sensors_calib"],
                        device_a["rgb_calib"],
                        depth_m,
                    )
                    cv2.circle(
                        rgb_image_a,
                        (int(gaze_projection[0]), int(gaze_projection[1])),
                        15,
                        (0, 255, 0),
                        -1,
                    )

                cv2.imshow(rgb_window_a, np.rot90(rgb_image_a, -1))

            if device_b["observer"].rgb_image is not None:
                rgb_image_b = cv2.cvtColor(device_b["observer"].rgb_image, cv2.COLOR_BGR2RGB)

                if device_b["observer"].eye_image is not None:
                    cv2.imshow(eye_window_b, device_b["observer"].eye_image)

                    # input size: 240x640
                    img = torch.tensor(device_b["observer"].eye_image, device=args.device)
                    preds, lower, upper = inference_model_b.predict(img)
                    preds = preds.detach().cpu().numpy()
                    lower = lower.detach().cpu().numpy()
                    upper = upper.detach().cpu().numpy()
                    value_mapping = {
                        "yaw": preds[0][0],
                        "pitch": preds[0][1],
                        "yaw_lower": lower[0][0],
                        "pitch_lower": lower[0][1],
                        "yaw_upper": upper[0][0],
                        "pitch_upper": upper[0][1],
                    }

                    eye_gaze = EyeGaze
                    eye_gaze.yaw = value_mapping["yaw"]
                    eye_gaze.pitch = value_mapping["pitch"]

                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze,
                        "camera-rgb",
                        device_b["sensors_calib"],
                        device_b["rgb_calib"],
                        depth_m,
                    )
                    cv2.circle(
                        rgb_image_b,
                        (int(gaze_projection[0]), int(gaze_projection[1])),
                        15,
                        (0, 255, 0),
                        -1,
                    )

                cv2.imshow(rgb_window_b, np.rot90(rgb_image_b, -1))

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    device_a["streaming_client"].unsubscribe()
    device_a["streaming_manager"].stop_streaming()
    device_a["device_client"].disconnect(device_a["device"])

    device_b["streaming_client"].unsubscribe()
    device_b["streaming_manager"].stop_streaming()
    device_b["device_client"].disconnect(device_b["device"])


if __name__ == "__main__":
    main()
