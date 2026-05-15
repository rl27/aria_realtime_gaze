import argparse
import sys
import random

import aria.sdk as aria

import cv2
import cv2.aruco as aruco
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
        help="Type of interface to use for streaming. Options are usb, wifi, or subscribe.",
        choices=["usb", "wifi", "subscribe"],
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
        "--device-ip", help="IP address to connect to the device over wifi"
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

    # 1. Create DeviceClient instance, setting the IP address if specified
    device_client = aria.DeviceClient()

    client_config = aria.DeviceClientConfig()
    if args.device_ip:
        client_config.ip_v4_address = args.device_ip
    device_client.set_client_config(client_config)

    # 2. Connect to the device
    device = device_client.connect()

    # 3. Retrieve the device streaming_manager and streaming_client
    streaming_manager = device.streaming_manager
    
    if args.streaming_interface == "subscribe":
        # Act like streaming_subscribe: create a separate StreamingClient to just listen
        streaming_client = aria.StreamingClient()
    else:
        streaming_client = streaming_manager.streaming_client

        # 4. Use a custom configuration for streaming
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = args.profile_name
        # Note: by default streaming uses Wifi
        if args.streaming_interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config

    # 5. Get sensors calibration
    sensors_calib_json = streaming_manager.sensors_calibration()
    sensors_calib = device_calibration_from_json_string(sensors_calib_json)
    rgb_calib = sensors_calib.get_camera_calib("camera-rgb")

    dst_calib = get_linear_camera_calibration(512, 512, 150, "camera-rgb")

    # 6. Start streaming if not just subscribing
    if args.streaming_interface != "subscribe":
        streaming_manager.start_streaming()

    # 7. Configure subscription to listen to Aria's RGB and eye track stream
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
    if args.streaming_interface == "subscribe":
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
    streaming_client.subscription_config = config

    # 8. Create and attach the visualizer and start listening to streaming data
    class StreamingClientObserver:
        def __init__(self):
            self.rgb_image = None
            self.eye_image = None

        def on_image_received(self, image: np.array, record: ImageDataRecord):
            if record.camera_id == aria.CameraId.Rgb:
                self.rgb_image = image
            if record.camera_id == aria.CameraId.EyeTrack:
                self.eye_image = image

    observer = StreamingClientObserver()
    streaming_client.set_streaming_client_observer(observer)
    streaming_client.subscribe()

    # 9. Render the streaming data until we close the window
    rgb_window = "RGB images"
    maze_window = "Maze Game"
    eye_window = "Eye images"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 800, 50)

    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 640, 240)
    cv2.setWindowProperty(eye_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window, 800, 600)

    # Setup the Maze window with a specific size
    vw, vh = 1600, 1000
    cv2.namedWindow(maze_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(maze_window, vw, vh)
    cv2.setWindowProperty(maze_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(maze_window, 50, 50)

    # Initialize ArUco dictionary and parameters
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    aruco_params = aruco.DetectorParameters()
    try:
        aruco_detector = aruco.ArucoDetector(aruco_dict, aruco_params)
    except AttributeError:
        # Fallback for older OpenCV versions
        aruco_detector = None
    
    # Generate 4 ArUco markers for the screen corners
    marker_size = 180
    margin = 30
    m0 = aruco.generateImageMarker(aruco_dict, 0, marker_size)
    m1 = aruco.generateImageMarker(aruco_dict, 1, marker_size)
    m2 = aruco.generateImageMarker(aruco_dict, 2, marker_size)
    m3 = aruco.generateImageMarker(aruco_dict, 3, marker_size)
    
    # Generate random maze background map
    def generate_random_maze(cols, rows):
        maze = np.ones((rows * 2 + 1, cols * 2 + 1), dtype=np.uint8)
        def carve(r, c):
            maze[r*2+1, c*2+1] = 0
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            random.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and maze[nr*2+1, nc*2+1] == 1:
                    maze[r*2+1 + dr, c*2+1 + dc] = 0
                    carve(nr, nc)
        old_limit = sys.getrecursionlimit()
        sys.setrecursionlimit(max(old_limit, cols*rows*10))
        carve(0, 0)
        sys.setrecursionlimit(old_limit)
        # Keep the start closed so the dot cannot exit the maze backwards
        # maze[1, 0] = 0
        maze[rows*2-1, cols*2] = 0 # end
        return maze

    maze_cols, maze_rows = 12, 8
    path_size = 60
    wall_size = 15

    def get_coord(idx):
        # Even indices are walls, odd indices are paths
        return (idx // 2) * (path_size + wall_size) + (idx % 2) * wall_size

    maze_w = get_coord(maze_cols * 2 + 1)
    maze_h = get_coord(maze_rows * 2 + 1)
    maze_start_x = (vw - maze_w) // 2
    maze_start_y = (vh - maze_h) // 2

    def gen_static_maze_frame():
        maze_grid = generate_random_maze(maze_cols, maze_rows)
        frame = np.ones((vh, vw, 3), dtype=np.uint8) * 255
        frame[margin:margin+marker_size, margin:margin+marker_size] = cv2.cvtColor(m0, cv2.COLOR_GRAY2BGR)
        frame[margin:margin+marker_size, vw-marker_size-margin:vw-margin] = cv2.cvtColor(m1, cv2.COLOR_GRAY2BGR)
        frame[vh-marker_size-margin:vh-margin, vw-marker_size-margin:vw-margin] = cv2.cvtColor(m2, cv2.COLOR_GRAY2BGR)
        frame[vh-marker_size-margin:vh-margin, margin:margin+marker_size] = cv2.cvtColor(m3, cv2.COLOR_GRAY2BGR)

        for r in range(maze_grid.shape[0]):
            for c in range(maze_grid.shape[1]):
                if maze_grid[r, c] == 1:
                    cx1 = maze_start_x + get_coord(c)
                    cy1 = maze_start_y + get_coord(r)
                    cx2 = maze_start_x + get_coord(c + 1)
                    cy2 = maze_start_y + get_coord(r + 1)
                    cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 0, 0), -1)

        start_text_pos = (maze_start_x - 120, maze_start_y + get_coord(1) + 40)
        end_text_pos = (maze_start_x + maze_w + 20, maze_start_y + get_coord(maze_rows * 2 - 1) + 40)
        cv2.putText(frame, "START", start_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "END", end_text_pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return frame

    static_maze_frame = gen_static_maze_frame()
    
    # 10. Set up inference model
    inference_model = infer.EyeGazeInference(f"{os.path.dirname(__file__)}/model/weights.pth",
                                             f"{os.path.dirname(__file__)}/model/config.yaml",
                                             args.device)
    depth_m = 1

    rgb_image = None
    gaze_point_on_screen = None
    last_H = None
    
    # Physics and smoothing state for the gaze dot
    smoothed_gaze_x = None
    smoothed_gaze_y = None
    dot_x = float(maze_start_x + get_coord(1) + path_size / 2)
    dot_y = float(maze_start_y + get_coord(1) + path_size / 2)
    dot_vx = 0.0
    dot_vy = 0.0
    dot_radius = 12

    # Calibration state
    is_calibrating = False
    is_calibrated = False
    calibration_step = 0
    calibration_points = [
        (vw // 2, vh // 2),
        (200, 200),
        (vw - 200, 200),
        (vw - 200, vh - 200),
        (200, vh - 200)
    ]
    calib_wait_frames = 45
    calib_collect_frames = 30
    current_calib_frame = 0
    collected_gaze = []
    calibration_data = [] # List of (target_pt, [gaze_pts...])
    calib_transform = None

    with ctrl_c_handler() as ctrl_c:
        while not ctrl_c:
            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break
            elif key == ord("c"):
                is_calibrating = True
                is_calibrated = False
                calibration_step = 0
                current_calib_frame = 0
                calibration_data = []
                collected_gaze = []

            # Start fresh frame with pre-computed maze and markers
            maze_frame = static_maze_frame.copy()

            if observer.rgb_image is not None:
                # Undistort the original image first
                undistorted_rgb = distort_by_calibration(observer.rgb_image, dst_calib, rgb_calib)
                
                # Rotate because typical camera mount might require it
                rgb_image = np.ascontiguousarray(np.rot90(cv2.cvtColor(undistorted_rgb, cv2.COLOR_BGR2RGB), -1))

                if observer.eye_image is not None:
                    # input size: 240x640
                    img = torch.tensor(observer.eye_image, device=args.device)
                    preds, lower, upper = inference_model.predict(img)
                    preds = preds.detach().cpu().numpy()
                    
                    eye_gaze = EyeGaze
                    eye_gaze.yaw = preds[0][0]
                    eye_gaze.pitch = preds[0][1]

                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze,
                        "camera-rgb",
                        sensors_calib,
                        dst_calib,
                        depth_m,
                    )
                    
                    # Compute gaze on the unrotated camera feed, then rotate for our drawing
                    gx, gy = int(gaze_projection[0]), int(gaze_projection[1])
                    
                    # Original image was WxH. It was rotated -90 (clockwise), so new shape is HxW.
                    # Clockwise rotation mapping: x_new = H - 1 - y, y_new = x
                    original_h = undistorted_rgb.shape[0]
                    rotated_gx = original_h - 1 - gy
                    rotated_gy = gx
                    
                    cv2.circle(rgb_image, (rotated_gx, rotated_gy), 5, (0, 255, 0), -1)

                    # ArUco marker detection to map gaze to screen
                    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
                    if aruco_detector is not None:
                        corners, ids, rejected = aruco_detector.detectMarkers(gray)
                    else:
                        corners, ids, rejected = aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

                    if ids is not None:
                        # Highlight detected markers on the RGB image
                        if aruco_detector is not None:
                            cv2.aruco.drawDetectedMarkers(rgb_image, corners, ids)
                        else:
                            aruco.drawDetectedMarkers(rgb_image, corners, ids)
                        
                        # Map markers found to the theoretical corners they represent on the maze frame
                        screen_pts = []
                        camera_pts = []
                        for i in range(len(ids)):
                            marker_id = ids[i][0]
                            if marker_id in [0, 1, 2, 3]:
                                # Center of marker in camera
                                c = np.mean(corners[i][0], axis=0)
                                camera_pts.append(c)
                                
                                # Center of marker on screen (adjusted for margin)
                                if marker_id == 0: screen_pts.append([margin + marker_size/2, margin + marker_size/2])
                                elif marker_id == 1: screen_pts.append([vw - margin - marker_size/2, margin + marker_size/2])
                                elif marker_id == 2: screen_pts.append([vw - margin - marker_size/2, vh - margin - marker_size/2])
                                elif marker_id == 3: screen_pts.append([margin + marker_size/2, vh - margin - marker_size/2])

                        if len(camera_pts) == 4:
                            camera_pts = np.array(camera_pts, dtype=np.float32)
                            screen_pts = np.array(screen_pts, dtype=np.float32)

                            # Calculate Homography matrix
                            H, _ = cv2.findHomography(camera_pts, screen_pts)
                            if H is not None:
                                last_H = H

                    if last_H is not None:
                        gaze_pt = np.array([[[rotated_gx, rotated_gy]]], dtype=np.float32)
                        transformed_gaze = cv2.perspectiveTransform(gaze_pt, last_H)
                        gaze_point_on_screen = transformed_gaze[0][0]

            # Draw the gaze dot on the screen if it falls within the screen boundaries
            if gaze_point_on_screen is not None:
                raw_cx, raw_cy = float(gaze_point_on_screen[0]), float(gaze_point_on_screen[1])
                
                if is_calibrating:
                    # Hide the maze, leave markers visible
                    cv2.rectangle(maze_frame, (maze_start_x - 130, maze_start_y - 20), (maze_start_x + maze_w + 130, maze_start_y + maze_h + 20), (255, 255, 255), -1)

                    # Draw calibration target
                    target_pt = calibration_points[calibration_step]
                    cv2.circle(maze_frame, target_pt, 15, (0, 0, 255), -1)
                    
                    # Highlight inner part based on progress
                    if current_calib_frame > calib_wait_frames:
                        cv2.circle(maze_frame, target_pt, 5, (0, 255, 0), -1)
                        
                    cv2.putText(maze_frame, "Look at the red dot and keep your head still", (vw//2 - 300, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(maze_frame, f"Point {calibration_step + 1}/{len(calibration_points)}", (vw//2 - 80, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    current_calib_frame += 1
                    
                    if current_calib_frame > calib_wait_frames:
                        collected_gaze.append([raw_cx, raw_cy])
                        
                    if current_calib_frame > calib_wait_frames + calib_collect_frames:
                        calibration_data.append((calibration_points[calibration_step], collected_gaze))
                        calibration_step += 1
                        current_calib_frame = 0
                        collected_gaze = []
                        if calibration_step >= len(calibration_points):
                            is_calibrating = False
                            is_calibrated = True
                            # Compute perspective transformation fixing screen mapping
                            src_pts = []
                            dst_pts = []
                            for target, gazes in calibration_data:
                                if len(gazes) > 0:
                                    avg_gaze = np.mean(gazes, axis=0)
                                    src_pts.append(avg_gaze)
                                    dst_pts.append(target)
                            
                            if len(src_pts) == len(calibration_points):
                                src_pts = np.array(src_pts, dtype=np.float32)
                                dst_pts = np.array(dst_pts, dtype=np.float32)
                                calib_transform, _ = cv2.findHomography(src_pts, dst_pts)
                                
                                # Reset maze and dot position
                                static_maze_frame = gen_static_maze_frame()
                                dot_x = float(maze_start_x + get_coord(1) + path_size / 2)
                                dot_y = float(maze_start_y + get_coord(1) + path_size / 2)
                                dot_vx = 0.0
                                dot_vy = 0.0
                            else:
                                is_calibrated = False # Failed
                
                if is_calibrated and calib_transform is not None:
                    pt = np.array([[[raw_cx, raw_cy]]], dtype=np.float32)
                    transformed_pt = cv2.perspectiveTransform(pt, calib_transform)
                    raw_cx, raw_cy = float(transformed_pt[0][0][0]), float(transformed_pt[0][0][1])

                # 1. EMA to smooth the raw gaze coordinates
                if smoothed_gaze_x is None:
                    smoothed_gaze_x, smoothed_gaze_y = raw_cx, raw_cy
                else:
                    alpha = 0.7  # Increased to make the raw tracking target more responsive
                    smoothed_gaze_x = alpha * raw_cx + (1 - alpha) * smoothed_gaze_x
                    smoothed_gaze_y = alpha * raw_cy + (1 - alpha) * smoothed_gaze_y
                
                # Only move the dot if gaze actively falls onto the screen tracking boundaries (with an expanded margin)
                margin_expand = 400
                if -margin_expand <= smoothed_gaze_x < vw + margin_expand and -margin_expand <= smoothed_gaze_y < vh + margin_expand:
                    spring_k = 0.015  # Decreased to lower acceleration
                    damping = 0.7     # Decreased multiplier for higher friction/lower top speed
                    
                    ax = (smoothed_gaze_x - dot_x) * spring_k
                    ay = (smoothed_gaze_y - dot_y) * spring_k
                    
                    dot_vx = (dot_vx + ax) * damping
                    dot_vy = (dot_vy + ay) * damping

                    # Maze Collision Logic: Sample background pixels to see if moving there hits a black wall
                    def check_collision(nx, ny):
                        perimeter_angles = [0, np.pi/2, np.pi, 3*np.pi/2, np.pi/4, 3*np.pi/4, 5*np.pi/4, 7*np.pi/4]
                        for a in perimeter_angles:
                            px = int(nx + dot_radius * np.cos(a))
                            py = int(ny + dot_radius * np.sin(a))
                            if 0 <= px < vw and 0 <= py < vh:
                                if np.all(static_maze_frame[py, px] == [0, 0, 0]):
                                    return True
                            else:
                                return True # Hit outer window boundary
                        return False

                    # Resolve X and Y independently with sub-stepping to prevent tunneling through thin walls
                    steps_x = int(abs(dot_vx)) + 1
                    step_dx = dot_vx / steps_x
                    for _ in range(steps_x):
                        if not check_collision(dot_x + step_dx, dot_y):
                            dot_x += step_dx
                        else:
                            dot_vx = 0
                            break
                            
                    steps_y = int(abs(dot_vy)) + 1
                    step_dy = dot_vy / steps_y
                    for _ in range(steps_y):
                        if not check_collision(dot_x, dot_y + step_dy):
                            dot_y += step_dy
                        else:
                            dot_vy = 0
                            break

            # Draw the dot itself, mapped natively onto the canvas if not calibrating
            if not is_calibrating:
                draw_x = int(max(0, min(vw - 1, dot_x)))
                draw_y = int(max(0, min(vh - 1, dot_y)))
                cv2.circle(maze_frame, (draw_x, draw_y), dot_radius, (255, 0, 0), -1)

            cv2.imshow(maze_window, maze_frame)
            if rgb_image is not None:
                cv2.imshow(rgb_window, rgb_image)
            if observer.eye_image is not None:
                cv2.imshow(eye_window, observer.eye_image)

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    if args.streaming_interface != "subscribe":
        streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
