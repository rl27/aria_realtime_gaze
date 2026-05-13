import argparse
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np

from gaze_correction import apply_affine_correction


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
STATE_DIR = RUNTIME / "state"

PLAYER_STYLES = {
    "p1": {
        "color": (255, 80, 40),
        "label": "p1",
    },
    "p2": {
        "color": (40, 220, 80),
        "label": "p2",
    },
}


class CursorPhysics:
    """Smooth inertia-based cursor that stays clamped to the screen."""

    def __init__(self, smoothing=0.12, damping=0.7):
        self.pos = None       # current rendered position [x, y]
        self.vel = np.zeros(2, dtype=np.float64)  # pixels/frame
        self.smoothing = smoothing   # spring stiffness (0=frozen, 1=instant)
        self.damping = damping       # velocity retention per frame (0=no inertia, 1=no friction)
        self.last_time = None
        self.last_active_time = None  # last time fresh gaze data arrived
        self.stale = False

    def update(self, target_xy, screen_w, screen_h):
        target = np.array(target_xy, dtype=np.float64)
        now = time.time()
        self.last_active_time = now
        self.stale = False

        if self.pos is None:
            self.pos = target.copy()
            self.last_time = now
            self._clamp(screen_w, screen_h)
            return self.pos.tolist()

        dt = now - self.last_time
        self.last_time = now
        dt_scale = min(dt * 30.0, 3.0)

        diff = target - self.pos
        accel = diff * self.smoothing * dt_scale
        self.vel = self.vel * (self.damping ** dt_scale) + accel
        self.pos += self.vel * dt_scale

        self._clamp(screen_w, screen_h)
        return self.pos.tolist()

    def coast(self, screen_w, screen_h):
        """Keep the cursor drifting with decaying velocity when data is stale."""
        self.stale = True
        if self.pos is None:
            return None

        now = time.time()
        dt = now - (self.last_time or now)
        self.last_time = now
        dt_scale = min(dt * 30.0, 3.0)

        self.vel *= self.damping ** dt_scale
        self.pos += self.vel * dt_scale
        self._clamp(screen_w, screen_h)
        return self.pos.tolist()

    def stale_seconds(self):
        if self.last_active_time is None:
            return 0.0
        return time.time() - self.last_active_time

    def _clamp(self, w, h):
        self.pos[0] = np.clip(self.pos[0], 0, w - 1)
        self.pos[1] = np.clip(self.pos[1], 0, h - 1)


_cursors: dict[str, CursorPhysics] = {}

# How long to keep showing cursor after data goes stale before hiding entirely
_STALE_HOLD_SEC = 3.0
# Pulse frequency in Hz
_PULSE_HZ = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="runtime/tv_scene.png")
    p.add_argument("--state-dir", default="runtime/state")
    p.add_argument("--corrections", default="runtime/gaze_corrections.json")
    p.add_argument("--window-name", default="Aria I Spy TV")
    p.add_argument("--max-age-ms", type=int, default=750)
    p.add_argument("--dot-radius", type=int, default=24)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--no-fullscreen", action="store_true")
    p.add_argument("--once", action="store_true", help="Render one frame and exit.")
    p.add_argument("--output", default=None, help="Optional output image for --once.")
    return p.parse_args()


def resolve_app_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def read_json(path, default=None):
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def valid_player_state(state, max_age_ms):
    if not state:
        return False

    if state.get("source") != "aria":
        return False

    if state.get("forbidden_webcam_used") is True:
        return False

    if state.get("homography_ok") is not True:
        return False

    if state.get("gaze_tv") is None:
        return False

    timestamp_ms = state.get("timestamp_unix_ms")
    if timestamp_ms is None:
        return False

    age_ms = int(time.time() * 1000) - int(timestamp_ms)
    return age_ms <= max_age_ms


def draw_player_dot(frame, player_id, tv_xy, radius, opacity=1.0):
    if opacity <= 0.01:
        return

    style = PLAYER_STYLES[player_id]
    x = int(round(tv_xy[0]))
    y = int(round(tv_xy[1]))
    color = style["color"]
    label = style["label"]

    h, w = frame.shape[:2]
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    if opacity >= 0.99:
        cv2.circle(frame, (x, y), radius + 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(frame, (x, y), radius, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        label_pos = (x + radius + 10, y + radius + 10)
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    else:
        overlay = frame.copy()
        cv2.circle(overlay, (x, y), radius + 5, (255, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay, (x, y), radius, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        label_pos = (x + radius + 10, y + radius + 10)
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 5, cv2.LINE_AA)
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
        cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, frame)


def _pulse_opacity(stale_sec):
    """Sinusoidal pulse that fades out over _STALE_HOLD_SEC."""
    fade = max(0.0, 1.0 - stale_sec / _STALE_HOLD_SEC)
    pulse = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(stale_sec * _PULSE_HZ * 2.0 * math.pi))
    return fade * pulse


def render_calibration_overlay(frame, target_tv, point_name, player_id, layout):
    """Draw a crosshair on top of the scene with the I Spy image blacked out."""
    if layout and layout.get("image_rect_tv"):
        x0, y0, x1, y1 = [int(v) for v in layout["image_rect_tv"]]
        cv2.rectangle(frame, (x0, y0), (x1, y1), (30, 30, 30), -1)

    tx, ty = int(round(target_tv[0])), int(round(target_tv[1]))

    arm = 40
    cv2.line(frame, (tx - arm, ty), (tx + arm, ty), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.line(frame, (tx, ty - arm), (tx, ty + arm), (0, 0, 0), 7, cv2.LINE_AA)
    cv2.line(frame, (tx - arm, ty), (tx + arm, ty), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.line(frame, (tx, ty - arm), (tx, ty + arm), (255, 255, 255), 3, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 18, (0, 200, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 18, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (tx, ty), 4, (255, 255, 255), -1, cv2.LINE_AA)

    label = f"[{player_id}] Look here: {point_name}"
    cv2.putText(frame, label, (tx + 30, ty - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(frame, label, (tx + 30, ty - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)


def render_frame(base, state_dir, corrections_path, max_age_ms, dot_radius):
    h, w = base.shape[:2]

    frame = base.copy()

    cal = read_json(state_dir / "calibration.json")
    if cal and cal.get("active") and cal.get("target_tv"):
        layout = read_json(ROOT / "runtime" / "tv_layout.json")
        render_calibration_overlay(
            frame, cal["target_tv"], cal.get("point_name", ""), cal.get("player", ""),
            layout,
        )
        return frame

    corrections = read_json(corrections_path, default={})

    for player_id in PLAYER_STYLES:
        if player_id not in _cursors:
            _cursors[player_id] = CursorPhysics()

        cursor = _cursors[player_id]
        state = read_json(state_dir / f"{player_id}.json")

        if valid_player_state(state, max_age_ms):
            tv_xy = apply_affine_correction(
                state["gaze_tv"],
                corrections.get(player_id),
            )
            smoothed_xy = cursor.update(tv_xy, w, h)
            draw_player_dot(frame, player_id, smoothed_xy, dot_radius)
        else:
            # Data stale — coast with decaying velocity and pulse opacity
            coasted_xy = cursor.coast(w, h)
            if coasted_xy is not None:
                stale_sec = cursor.stale_seconds()
                if stale_sec < _STALE_HOLD_SEC:
                    opacity = _pulse_opacity(stale_sec)
                    draw_player_dot(frame, player_id, coasted_xy, dot_radius, opacity)

    return frame


def main():
    args = parse_args()
    scene_path = resolve_app_path(args.scene)
    state_dir = resolve_app_path(args.state_dir)
    corrections_path = resolve_app_path(args.corrections)

    base = cv2.imread(str(scene_path), cv2.IMREAD_COLOR)
    if base is None:
        raise SystemExit(
            f"Missing or unreadable {scene_path}. Run `python make_tv_scene.py` first."
        )

    if args.once:
        frame = render_frame(
            base,
            state_dir,
            corrections_path,
            args.max_age_ms,
            args.dot_radius,
        )
        if args.output:
            output_path = resolve_app_path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), frame)
        return

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    if not args.no_fullscreen:
        cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    delay_ms = max(1, int(1000 / args.fps))

    while True:
        frame = render_frame(
            base,
            state_dir,
            corrections_path,
            args.max_age_ms,
            args.dot_radius,
        )

        cv2.imshow(args.window_name, frame)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key in (27, ord("q")):
            break

    cv2.destroyWindow(args.window_name)


if __name__ == "__main__":
    main()
