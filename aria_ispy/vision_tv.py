import json
from typing import Dict

import cv2
import numpy as np


def load_layout(path):
    with open(path, "r") as f:
        return json.load(f)


def make_apriltag_detector():
    """
    Use OpenCV AprilTag support.
    The generated TV scene must use this same dictionary.
    """

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.adaptiveThreshWinSizeMin = 3
    params.adaptiveThreshWinSizeMax = 53
    params.adaptiveThreshWinSizeStep = 4
    params.minMarkerPerimeterRate = 0.01
    params.polygonalApproxAccuracyRate = 0.08

    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_apriltags(frame_bgr, detector) -> Dict[int, np.ndarray]:
    """
    Return:
      {
        tag_id: np.array([
          [x_tl, y_tl],
          [x_tr, y_tr],
          [x_br, y_br],
          [x_bl, y_bl],
        ], dtype=np.float32)
      }

    Coordinates are in the raw Aria RGB frame.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)

    if ids is None:
        return {}

    out = {}

    for marker_corners, marker_id in zip(corners, ids.flatten()):
        out[int(marker_id)] = marker_corners[0].astype(np.float32)

    return out


def estimate_rgb_to_tv_homography(detections, layout):
    """
    Estimate H such that:
      TV pixel = H * Aria RGB pixel

    Each AprilTag gives 4 matching points.
    Two tags gives 8 points and is the minimum.
    Four tags is preferred.
    """

    rgb_pts = []
    tv_pts = []

    tag_corners_tv = layout["tag_corners_tv"]

    for tag_id_str, tv_corners in tag_corners_tv.items():
        tag_id = int(tag_id_str)

        if tag_id not in detections:
            continue

        rgb_corners = detections[tag_id]
        tv_corners = np.array(tv_corners, dtype=np.float32)

        for rgb_pt, tv_pt in zip(rgb_corners, tv_corners):
            rgb_pts.append(rgb_pt)
            tv_pts.append(tv_pt)

    debug = {
        "tags_found": sorted(list(detections.keys())),
        "num_points": len(rgb_pts),
    }

    if len(rgb_pts) < 8:
        debug["reason"] = "Need at least two complete AprilTags"
        return None, debug

    rgb_pts = np.array(rgb_pts, dtype=np.float32)
    tv_pts = np.array(tv_pts, dtype=np.float32)

    H, inliers = cv2.findHomography(
        rgb_pts,
        tv_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if H is None:
        debug["reason"] = "cv2.findHomography failed"
        return None, debug

    debug["inliers"] = int(inliers.sum()) if inliers is not None else None
    return H, debug


def map_point_homography(H, xy):
    """
    Map one Aria RGB pixel to TV pixel.
    """

    pt = np.array([[[xy[0], xy[1]]]], dtype=np.float32)
    mapped = cv2.perspectiveTransform(pt, H)
    return [float(mapped[0, 0, 0]), float(mapped[0, 0, 1])]


def tv_to_ispy(tv_xy, layout):
    """
    Convert TV pixel coordinate to the displayed I Spy image coordinate.

    Returns None if gaze is outside the displayed I Spy image.
    """

    if tv_xy is None:
        return None

    tv_x, tv_y = tv_xy
    x0, y0, x1, y1 = layout["image_rect_tv"]
    img_w, img_h = layout["ispy_image_size_px"]

    if not (x0 <= tv_x <= x1 and y0 <= tv_y <= y1):
        return None

    u = (tv_x - x0) / (x1 - x0)
    v = (tv_y - y0) / (y1 - y0)

    return [float(u * img_w), float(v * img_h)]


def rect_contains_point(xy, xyxy, radius_px=0):
    if xy is None:
        return False

    x, y = xy
    x0, y0, x1, y1 = xyxy

    return (
        x0 - radius_px <= x <= x1 + radius_px
        and y0 - radius_px <= y <= y1 + radius_px
    )
