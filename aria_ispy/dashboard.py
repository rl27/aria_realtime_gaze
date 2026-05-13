import json
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_autorefresh import st_autorefresh

from gaze_correction import apply_affine_correction
from vision_tv import rect_contains_point, tv_to_ispy


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
STATE_DIR = RUNTIME / "state"
LAYOUT_PATH = RUNTIME / "tv_layout.json"
ROIS_PATH = RUNTIME / "rois.json"
ROIS_DIR = RUNTIME / "rois"
GAZE_CORRECTIONS_PATH = RUNTIME / "gaze_corrections.json"
GAZE_SAMPLES_PATH = RUNTIME / "gaze_calibration_samples.json"
CALIBRATION_SIGNAL_PATH = STATE_DIR / "calibration.json"
ASSETS_DIR = ROOT / "assets"

RUNTIME.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

if "app_initialized" not in st.session_state:
    st.session_state["app_initialized"] = True
    if CALIBRATION_SIGNAL_PATH.exists():
        CALIBRATION_SIGNAL_PATH.unlink()

PLAYERS = ["p1", "p2"]




def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default

    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f"{path.stem}_tmp_{os.getpid()}_{uuid.uuid4().hex}{path.suffix}"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def list_ispy_images():
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(
        path for path in ASSETS_DIR.iterdir() if path.is_file() and path.suffix.lower() in exts
    )


def rois_path_for_image(image_path):
    return ROIS_DIR / f"{image_path.stem}.json"


def read_rois_for_image(image_path):
    image_rois_path = rois_path_for_image(image_path)
    if image_rois_path.exists():
        return read_json(image_rois_path, default={})
    return read_json(ROIS_PATH, default={})


def write_rois_for_image(image_path, rois):
    write_json(rois_path_for_image(image_path), rois)
    # Keep the legacy/current ROI file updated for quick inspection.
    write_json(ROIS_PATH, rois)


def load_player_state(player_id):
    return read_json(STATE_DIR / f"{player_id}.json")


def load_player_frame(state):
    if not state or "frame_path" not in state:
        return None

    frame_path = Path(state["frame_path"])
    if not frame_path.exists():
        return None

    bgr = cv2.imread(str(frame_path))
    if bgr is None:
        return None

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def set_calibration_target(player_id, point_name, target_tv):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CALIBRATION_SIGNAL_PATH, {
        "active": True,
        "player": player_id,
        "point_name": point_name,
        "target_tv": target_tv,
    })


def clear_calibration_target():
    write_json(CALIBRATION_SIGNAL_PATH, {"active": False})


def persist_samples():
    write_json(GAZE_SAMPLES_PATH, st.session_state["gaze_calibration_samples"])


def calibration_points(layout, mode):
    """Generate calibration grid points that avoid the AprilTag corners."""
    tv_w = layout["tv_width"]
    tv_h = layout["tv_height"]
    tag_size = layout.get("tag_size_px", 250)
    margin = 24
    pad = 30

    safe_left = margin + tag_size + pad
    safe_top = margin + tag_size + pad
    safe_right = tv_w - margin - tag_size - pad
    safe_bottom = tv_h - margin - tag_size - pad
    cx = tv_w / 2.0
    cy = tv_h / 2.0

    if mode == "9-point":
        xs = [safe_left, cx, safe_right]
        ys = [safe_top, cy, safe_bottom]
        labels = [
            ["top_left", "top_center", "top_right"],
            ["mid_left", "center", "mid_right"],
            ["bot_left", "bot_center", "bot_right"],
        ]
        return [
            (labels[iy][ix], [xs[ix], ys[iy]])
            for iy in range(3)
            for ix in range(3)
        ]
    else:
        return [
            ("center", [cx, cy]),
            ("top_left", [safe_left, safe_top]),
            ("top_right", [safe_right, safe_top]),
            ("bottom_left", [safe_left, safe_bottom]),
            ("bottom_right", [safe_right, safe_bottom]),
        ]


def fit_affine_correction(samples):
    src = []
    dst = []

    for sample in samples:
        if sample.get("measured_tv") is None or sample.get("target_tv") is None:
            continue
        src.append(sample["measured_tv"])
        dst.append(sample["target_tv"])

    if len(src) < 3:
        return None

    src = np.array(src, dtype=np.float64)
    dst = np.array(dst, dtype=np.float64)
    design = np.column_stack([src[:, 0], src[:, 1], np.ones(len(src))])
    ax, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
    ay, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)

    return [
        [float(ax[0]), float(ax[1]), float(ax[2])],
        [float(ay[0]), float(ay[1]), float(ay[2])],
    ]


def get_corrected_gaze(state, layout, player_id):
    if not state:
        return None, None

    correction = st.session_state.get("gaze_corrections", {}).get(player_id)
    corrected_tv = apply_affine_correction(state.get("gaze_tv"), correction)
    corrected_ispy = tv_to_ispy(corrected_tv, layout) if layout else state.get("gaze_ispy")
    return corrected_tv, corrected_ispy


def draw_game_overlay(ispy_img, rois, selected_roi_name, player_states, layout):
    img = ispy_img.copy().convert("RGB")
    draw = ImageDraw.Draw(img)

    roi = rois.get(selected_roi_name)
    if roi and roi.get("type") == "rect":
        x0, y0, x1, y1 = roi["xyxy"]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=5)

    colors = {"p1": "blue", "p2": "green"}
    for player_id, state in player_states.items():
        _tv_xy, gaze = get_corrected_gaze(state, layout, player_id)
        if gaze is None:
            continue

        x, y = gaze
        r = 14
        color = colors.get(player_id, "blue")
        draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=5)
        draw.text((x + 16, y + 16), player_id, fill=color)

    return img


def update_dwell(player_id, hit, threshold_ms):
    now_ms = int(time.time() * 1000)
    key_start = f"{player_id}_dwell_start_ms"
    key_found = f"{player_id}_found"

    st.session_state.setdefault(key_start, None)
    st.session_state.setdefault(key_found, False)

    if st.session_state[key_found]:
        return True, threshold_ms

    if hit:
        if st.session_state[key_start] is None:
            st.session_state[key_start] = now_ms

        elapsed = now_ms - st.session_state[key_start]
        if elapsed >= threshold_ms:
            st.session_state[key_found] = True
            return True, elapsed

        return False, elapsed

    st.session_state[key_start] = None
    return False, 0


def reset_round():
    for player_id in PLAYERS:
        st.session_state[f"{player_id}_dwell_start_ms"] = None
        st.session_state[f"{player_id}_found"] = False

    st.session_state["round_start_ms"] = int(time.time() * 1000)
    st.session_state["winner"] = None
    st.session_state["win_time_s"] = None


def require_ispy_image(image_path):
    if not image_path.exists():
        st.error(f"Missing {image_path}.")
        st.stop()
    return Image.open(image_path).convert("RGB")


st.set_page_config(page_title="Aria I Spy", layout="wide")
st.title("Aria I Spy Dashboard")

layout = read_json(LAYOUT_PATH)
available_images = list_ispy_images()

if not available_images:
    st.error("No I Spy images found. Add JPG or PNG files to aria_ispy/assets/.")
    st.stop()

layout_image_name = layout.get("source_image_name") if layout else None
default_image_idx = 0
if layout_image_name:
    for idx, image_path in enumerate(available_images):
        if image_path.name == layout_image_name:
            default_image_idx = idx
            break

selected_image_path = st.sidebar.selectbox(
    "I Spy image",
    available_images,
    index=default_image_idx,
    format_func=lambda path: path.name,
)

if layout_image_name and selected_image_path.name != layout_image_name:
    st.sidebar.warning(
        "Selected image does not match runtime/tv_layout.json. "
        f"Regenerate the TV scene for {selected_image_path.name} before playing."
    )

st.sidebar.code(
    f"python make_tv_scene.py --image assets/{selected_image_path.name}",
    language="bash",
)

rois = read_rois_for_image(selected_image_path)
ispy_img = require_ispy_image(selected_image_path)
st.session_state.setdefault(
    "gaze_calibration_samples",
    read_json(GAZE_SAMPLES_PATH, default={p: [] for p in PLAYERS}),
)
for _p in PLAYERS:
    st.session_state["gaze_calibration_samples"].setdefault(_p, [])

st.session_state.setdefault(
    "gaze_corrections",
    read_json(GAZE_CORRECTIONS_PATH, default={}),
)

tab_calib, tab_game = st.tabs(
    [
        "1. TV registration + gaze calibration",
        "2. Game",
    ]
)


with tab_calib:
    st.header("TV registration + gaze calibration")

    if layout is None:
        st.warning("Missing aria_ispy/runtime/tv_layout.json. Run `python make_tv_scene.py`.")
    else:
        st.json(
            {
                "tv_width": layout.get("tv_width"),
                "tv_height": layout.get("tv_height"),
                "image_rect_tv": layout.get("image_rect_tv"),
                "ispy_image_size_px": layout.get("ispy_image_size_px"),
                "displayed_image_size_tv_px": layout.get("displayed_image_size_tv_px"),
                "source_image_name": layout.get("source_image_name"),
                "tag_gap_px": layout.get("tag_gap_px"),
            }
        )

    if st.checkbox("Auto-refresh registration view", value=True):
        st_autorefresh(interval=200, key="registration_refresh")

    cols = st.columns(2)
    for col, player_id in zip(cols, PLAYERS):
        with col:
            st.subheader(player_id)
            state = load_player_state(player_id)
            frame_rgb = load_player_frame(state)

            if state is None:
                st.warning(f"No state yet for {player_id}. Is aria_bridge.py running?")
                continue

            if state.get("source") != "aria":
                st.error(f"{player_id}: invalid source {state.get('source')!r}; expected 'aria'.")
            if state.get("forbidden_webcam_used") is True:
                st.error(f"{player_id}: forbidden webcam path was used.")

            if frame_rgb is not None:
                st.image(frame_rgb, caption=f"{player_id} latest Aria RGB debug frame")

            age_ms = int(time.time() * 1000) - state.get("timestamp_unix_ms", 0)
            corrected_tv, corrected_ispy = get_corrected_gaze(state, layout, player_id)

            with st.expander("State details"):
                st.metric("state age ms", age_ms)
                st.write("source:", state.get("source"))
                st.write("tags_found:", state.get("tags_found"))
                st.write("homography_ok:", state.get("homography_ok"))
                st.write("gaze_rgb:", state.get("gaze_rgb"))
                st.write("gaze_tv:", state.get("gaze_tv"))
                st.write("gaze_ispy:", state.get("gaze_ispy"))
                st.write("corrected_gaze_tv:", corrected_tv)
                st.write("corrected_gaze_ispy:", corrected_ispy)

    st.divider()
    st.subheader("Player gaze correction")
    st.write(
        "This fits a per-player affine correction from measured TV gaze to known TV points. "
        "A crosshair will appear on the TV for each calibration point."
    )

    if layout is None:
        st.info("Generate the TV layout before recording gaze correction samples.")
    else:
        player_for_cal = st.selectbox("Player", PLAYERS, key="cal_player")
        mode = st.radio("Calibration grid", ["5-point", "9-point"], horizontal=True)
        points = calibration_points(layout, mode)

        st.session_state.setdefault("cal_point_idx", 0)
        st.session_state.setdefault("cal_active", False)

        samples = st.session_state["gaze_calibration_samples"][player_for_cal]
        recorded_names = {s["name"] for s in samples}

        btn_cols = st.columns(3)
        with btn_cols[0]:
            if st.button("Start calibration"):
                st.session_state["cal_active"] = True
                st.session_state["cal_point_idx"] = 0
                point_name, target = points[0]
                set_calibration_target(player_for_cal, point_name, target)
                st.rerun()

        with btn_cols[1]:
            if st.button("Stop calibration"):
                st.session_state["cal_active"] = False
                clear_calibration_target()
                st.rerun()

        with btn_cols[2]:
            if st.button("Clear all samples"):
                st.session_state["gaze_calibration_samples"][player_for_cal] = []
                st.session_state["gaze_corrections"].pop(player_for_cal, None)
                st.session_state["cal_active"] = False
                st.session_state["cal_point_idx"] = 0
                clear_calibration_target()
                persist_samples()
                write_json(GAZE_CORRECTIONS_PATH, st.session_state["gaze_corrections"])
                st.rerun()

        if st.session_state["cal_active"]:
            idx = st.session_state["cal_point_idx"]
            if idx < len(points):
                point_name, target = points[idx]
                set_calibration_target(player_for_cal, point_name, target)

                st.info(
                    f"Point {idx + 1}/{len(points)}: **{point_name}** — "
                    f"ask {player_for_cal} to look at the crosshair on the TV"
                )

                if st.button("Record sample", type="primary"):
                    state = load_player_state(player_for_cal)
                    if not state or state.get("gaze_tv") is None:
                        st.error("No gaze_tv data available. Is aria_bridge running?")
                    else:
                        st.session_state["gaze_calibration_samples"][player_for_cal].append(
                            {
                                "name": point_name,
                                "target_tv": target,
                                "measured_tv": state["gaze_tv"],
                            }
                        )
                        persist_samples()
                        st.session_state["cal_point_idx"] = idx + 1
                        if idx + 1 >= len(points):
                            clear_calibration_target()
                            st.session_state["cal_active"] = False
                        st.rerun()
            else:
                clear_calibration_target()
                st.session_state["cal_active"] = False

        progress_cols = st.columns(len(points))
        for i, (pname, _target) in enumerate(points):
            with progress_cols[i]:
                if pname in recorded_names:
                    st.success(pname)
                elif st.session_state.get("cal_active") and i == st.session_state.get("cal_point_idx"):
                    st.warning(pname)
                else:
                    st.caption(pname)

        correction = fit_affine_correction(samples)
        if correction is not None:
            st.session_state["gaze_corrections"][player_for_cal] = correction
            write_json(GAZE_CORRECTIONS_PATH, st.session_state["gaze_corrections"])
            persist_samples()
            st.success(f"Affine gaze correction fitted for {player_for_cal}.")

        with st.expander("Calibration details"):
            st.write("samples:", samples)
            st.write("affine_correction:", st.session_state["gaze_corrections"].get(player_for_cal))


with tab_game:
    st.header("Game")

    game_ready = True
    if layout is None:
        st.warning("Generate TV layout before using the game tab.")
        game_ready = False

    if not rois:
        st.warning(
            "No ROIs found. Run `python define_rois.py "
            f"--image {selected_image_path.name}` to create them, then reload this page."
        )
        game_ready = False

    if game_ready:
        roi_names = sorted(rois.keys())
        selected_object = st.selectbox("Object to find", roi_names) or roi_names[0]

        dwell_threshold_ms = st.slider("Dwell threshold ms", 200, 1500, 800, step=50)
        tolerance_px = st.slider("Tolerance px", 0, 120, 40, step=5)

        if st.button("Start / reset round"):
            reset_round()
            st.session_state["game_running"] = True

        if not st.session_state.get("game_running", False):
            st.info("Press **Start / reset round** to begin.")
            st.stop()

        game_over = st.session_state.get("winner") is not None

        if not game_over:
            st_autorefresh(interval=100, key="game_refresh")

        player_states = {player_id: load_player_state(player_id) for player_id in PLAYERS}
        roi = rois[selected_object]

        if game_over:
            elapsed_round_s = st.session_state.get("win_time_s", 0.0)
        else:
            elapsed_round_s = (
                int(time.time() * 1000) - st.session_state["round_start_ms"]
            ) / 1000.0

        st.metric("Round timer", f"{elapsed_round_s:.2f} s")

        status_cols = st.columns(2)
        for status_col, player_id in zip(status_cols, PLAYERS):
            state = player_states[player_id]
            _corrected_tv, corrected_ispy = get_corrected_gaze(state, layout, player_id)
            hit = rect_contains_point(corrected_ispy, roi["xyxy"], radius_px=tolerance_px)

            if game_over:
                found = st.session_state.get(f"{player_id}_found", False)
                dwell_elapsed = dwell_threshold_ms if found else 0
            else:
                found, dwell_elapsed = update_dwell(player_id, hit, dwell_threshold_ms)

            if found and st.session_state.get("winner") is None:
                st.session_state["winner"] = player_id
                st.session_state["win_time_s"] = elapsed_round_s
                st.rerun()

            with status_col:
                st.subheader(player_id)
                st.write("corrected_gaze_ispy:", corrected_ispy)
                st.write("inside target:", hit)
                st.progress(min(dwell_elapsed / dwell_threshold_ms, 1.0))
                st.metric("dwell ms", dwell_elapsed)
                if found:
                    st.success("FOUND")

        if game_over:
            st.success(f"Winner: {st.session_state['winner']} in {st.session_state['win_time_s']:.2f}s")

        overlay = draw_game_overlay(ispy_img, rois, selected_object, player_states, layout)
        st.image(overlay, caption=f"Find: {selected_object}", width="stretch")
