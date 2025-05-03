# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="device to run inference on",
    )
    return parser.parse_args()




import numba
from numba import jit, prange
import copy

shift = 10
skip = 5
width, height = (1408, 1408)
w, h = (400, 400)
doinference = True


lasthit = -1
totalhit = 0
black = np.zeros((height, width, 3), dtype=np.uint8)
lasthittarget = np.zeros((h + 1, w + 1, 3), dtype=np.int32)

images = []
gazes = []


@numba.jit(nopython=True, parallel=True, fastmath=True, cache=True, nogil=True)
def calcsads(targets):

    length = len(targets)
    sads = np.full((length, length), np.iinfo(np.int32).max, dtype=np.int32)
    
    for index in prange(length):
        target = targets[index]
        
        for sweepindex in prange(length):
            image = targets[sweepindex]

            if index != sweepindex and image.shape == target.shape:
                sad = np.sum(np.abs(np.subtract(image, target)))

                sads[index][sweepindex] = sad
                sads[sweepindex][index] = sad
        
    return np.argmax(np.bincount(sads.argmin(axis=1)))



model = torch.hub.load('ultralytics/yolov5', 'yolov5x6', pretrained=True)


object_window = "Object Tracking"
cv2.namedWindow(object_window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(object_window, 512, 512)
cv2.setWindowProperty(object_window, cv2.WND_PROP_TOPMOST, 1)
cv2.moveWindow(object_window, 50, 50)



def vote():
    global lasthit, black, lasthittarget, totalhit
    index = len(images) - 1
    row, col = gazes[index]
    
    image = images[index].astype(np.uint8, copy=True)
    image = cv2.rectangle(image, (col-w//2, row-h//2), (col+w//2, row+h//2), color=(0, 255, 0), thickness=2)
    cv2.putText(image, str(index), (0, 0+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)

    ###continue

    if index < lasthit + shift:
        image = cv2.hconcat([image, black])
        #@video.write(image)
        cv2.imshow(object_window, image)
        ###continue
        return
    
    targets = []

    for sweepindex in prange(lasthit + 1, index + 1):
        target = images[sweepindex].astype(np.int32, copy=True)
        targetrow, targetcol = gazes[sweepindex]
        targets.append(target[targetrow-h//2:targetrow+h//2+1, targetcol-w//2:targetcol+w//2+1])

    minsadindex = calcsads(targets)
    newhit = minsadindex + (lasthit + 1)
    lasthit = newhit

    newtargets = copy.deepcopy(targets)
    newtargets[minsadindex] = lasthittarget.copy()
    
    if True and calcsads(newtargets) == minsadindex:
        lasthittarget = targets[minsadindex].copy()
        image = cv2.hconcat([image, black])
        #@video.write(image)
        cv2.imshow(object_window, image)
        ###print(index)
        ###continue
        return
        pass
    else:
        pass

    if True and minsadindex < skip:
        image = cv2.hconcat([image, black])
        #@video.write(image)
        cv2.imshow(object_window, image)
        ###continue
        return
    else:
        lasthittarget = targets[minsadindex].copy()
        newhitimage = images[newhit].astype(np.uint8, copy=True)
        newhitrow, newhitcol = gazes[newhit]
        newhitmask = newhitimage.copy()
        newhitmask[newhitrow-h//2:newhitrow+h//2+1, newhitcol-w//2:newhitcol+w//2+1] = 0
        newhitimage = newhitimage - newhitmask
    
        if doinference == True:
            for idx, predict in model(targets[minsadindex].astype(np.uint8, copy=False)).pandas().xyxy[0].iterrows():
                newhitimage = cv2.rectangle(newhitimage, (newhitcol-w//2+round(predict["xmin"]), newhitrow-h//2+round(predict["ymin"])), (newhitcol-w//2+round(predict["xmax"]), newhitrow-h//2+round(predict["ymax"])), color=(0, 0, 255), thickness=2)
                cv2.putText(newhitimage, predict["name"], (newhitcol-w//2+round(predict["xmin"]), newhitrow-h//2+round(predict["ymin"])-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
        cv2.putText(newhitimage, str(newhit), (0, 0+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 1, cv2.LINE_AA)
        
        totalhit = totalhit + 1
    
        image = cv2.hconcat([image, newhitimage])
    
        black = newhitimage.copy()
        
        #@ideo.write(image)
        cv2.imshow(object_window, image)





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

    # 6. Start streaming
    streaming_manager.start_streaming()

    # 7. Configure subscription to listen to Aria's RGB and eye track stream
    config = streaming_client.subscription_config
    config.subscriber_data_type = aria.StreamingDataType.Rgb | aria.StreamingDataType.EyeTrack
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
    eye_window = "Eye tracking"

    cv2.namedWindow(rgb_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(rgb_window, 512, 512)
    cv2.setWindowProperty(rgb_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(rgb_window, 50, 50)

    cv2.namedWindow(eye_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(eye_window, 512, 512)
    cv2.setWindowProperty(eye_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(eye_window, 600, 50)

    # 10. Set up inference model
    inference_model = infer.EyeGazeInference(f"{os.path.dirname(__file__)}/model/weights.pth",
                                             f"{os.path.dirname(__file__)}/model/config.yaml",
                                             args.device)
    depth_m = 1


    rgb_image = None
    with ctrl_c_handler() as ctrl_c:
        while not (quit_keypress() or ctrl_c):
            if observer.rgb_image is not None:
                rgb_image = cv2.cvtColor(observer.rgb_image, cv2.COLOR_BGR2RGB)

                if observer.eye_image is not None:
                    cv2.imshow(eye_window, observer.eye_image)

                    # input size: 240x640
                    img = torch.tensor(observer.eye_image, device=args.device)
                    preds, lower, upper = inference_model.predict(img)
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

                    # Compute eye_gaze vector at depth_m reprojection in the image
                    # Top right = (0,0), down = +x, left = +y
                    gaze_projection = get_gaze_vector_reprojection(
                        eye_gaze,
                        "camera-rgb",
                        sensors_calib,
                        rgb_calib,
                        depth_m,
                    )
                    

                    ##images.append(rgb_image)
                    ##gazes.append(gaze_projection)
                    images.append(np.rot90(rgb_image, -1).copy())
                    gazes.append((int(gaze_projection[0]), width - 1 - int(gaze_projection[1])))
                    vote()


                    print(gaze_projection)
                    print(len(images), len(gazes))


                    cv2.circle(rgb_image, (int(gaze_projection[0]), int(gaze_projection[1])), 15, (0,255,0), -1)
                    

                cv2.imshow(rgb_window, np.rot90(rgb_image, -1))

    # 10. Unsubscribe from data and stop streaming
    print("Stop listening to image data")
    streaming_client.unsubscribe()
    streaming_manager.stop_streaming()
    device_client.disconnect(device)


if __name__ == "__main__":
    main()
