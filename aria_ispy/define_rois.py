"""Standalone OpenCV tool to define rectangular ROIs on an I Spy image.

Click and drag with the left mouse button to draw a rectangle. Release to
finish, then type a name into the terminal to save it. Press `d` to delete
an ROI by name, `q` or ESC to quit. ROIs are written to
`runtime/rois/<image_stem>.json` using the same schema the dashboard reads.
"""

import argparse
import json
import os
import sys
import uuid
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
ROIS_DIR = RUNTIME / "rois"
ROIS_LEGACY = RUNTIME / "rois.json"
ASSETS_DIR = ROOT / "assets"
LAYOUT_PATH = RUNTIME / "tv_layout.json"

WINDOW_NAME = "Define ROIs (drag to draw, s=save, d=delete, q=quit)"


def write_json_atomic(path: Path, data) -> None:
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


def read_json(path: Path, default):
    if not path.exists():
        return default
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def rois_path_for(image_path: Path) -> Path:
    return ROIS_DIR / f"{image_path.stem}.json"


def load_rois(image_path: Path) -> dict:
    return read_json(rois_path_for(image_path), default={})


def save_rois(image_path: Path, rois: dict) -> None:
    write_json_atomic(rois_path_for(image_path), rois)
    write_json_atomic(ROIS_LEGACY, rois)


def resolve_image(name: str | None) -> Path:
    if name:
        path = (ASSETS_DIR / name).resolve()
        if not path.exists():
            sys.exit(f"Image not found: {path}")
        return path

    layout = read_json(LAYOUT_PATH, default=None)
    if layout and layout.get("source_image_name"):
        candidate = ASSETS_DIR / layout["source_image_name"]
        if candidate.exists():
            return candidate

    images = sorted(p for p in ASSETS_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        sys.exit(f"No I Spy images found in {ASSETS_DIR}")
    return images[0]


class ROIEditor:
    def __init__(self, image_path: Path):
        self.image_path = image_path
        self.image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.image is None:
            sys.exit(f"Failed to read image: {image_path}")
        self.h, self.w = self.image.shape[:2]
        self.rois = load_rois(image_path)
        self.draft = None  # (x0, y0, x1, y1) live during drag
        self.last_saved = None  # (x0, y0, x1, y1) last committed rect
        self._drag_origin = None
        self._needs_redraw = True

    def _on_mouse(self, event, x, y, flags, _userdata):
        x = max(0, min(x, self.w - 1))
        y = max(0, min(y, self.h - 1))

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_origin = (x, y)
            self.draft = (x, y, x, y)
            self._needs_redraw = True
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_origin is not None:
            x0, y0 = self._drag_origin
            self.draft = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self._needs_redraw = True
        elif event == cv2.EVENT_LBUTTONUP and self._drag_origin is not None:
            x0, y0 = self._drag_origin
            self.draft = (min(x0, x), min(y0, y), max(x0, x), max(y0, y))
            self._drag_origin = None
            self._needs_redraw = True
            x0, y0, x1, y1 = self.draft
            if (x1 - x0) < 3 or (y1 - y0) < 3:
                print("[draft] rectangle too small, ignored")
                self.draft = None

    def _render(self):
        canvas = self.image.copy()
        for name, info in self.rois.items():
            xyxy = info.get("xyxy", [])
            if len(xyxy) != 4:
                continue
            x0, y0, x1, y1 = (int(v) for v in xyxy)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 200, 0), 2)
            label = name
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(canvas, (x0, y0 - th - 6), (x0 + tw + 6, y0), (0, 200, 0), -1)
            cv2.putText(canvas, label, (x0 + 3, y0 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        if self.draft is not None:
            x0, y0, x1, y1 = self.draft
            cv2.rectangle(canvas, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.putText(canvas, "press 's' to save", (x0, max(0, y0 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        hud = [
            f"image: {self.image_path.name}   rois: {len(self.rois)}",
            "drag = draw   s = save draft   d = delete   q/ESC = quit",
        ]
        for i, text in enumerate(hud):
            y = 20 + i * 22
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW_NAME, canvas)

    def _prompt(self, message: str) -> str:
        try:
            return input(message).strip()
        except EOFError:
            return ""

    def _save_draft(self):
        if self.draft is None:
            print("[save] no draft rectangle. Draw one first.")
            return
        name = self._prompt("Name for this ROI: ")
        if not name:
            print("[save] empty name, cancelled")
            return
        if name in self.rois:
            answer = self._prompt(f"'{name}' already exists. Overwrite? [y/N] ")
            if answer.lower() not in ("y", "yes"):
                print("[save] cancelled")
                return
        x0, y0, x1, y1 = (float(v) for v in self.draft)
        self.rois[name] = {"type": "rect", "xyxy": [x0, y0, x1, y1]}
        save_rois(self.image_path, self.rois)
        print(f"[save] '{name}' -> [{x0:.0f}, {y0:.0f}, {x1:.0f}, {y1:.0f}]"
              f"  ({rois_path_for(self.image_path)})")
        self.last_saved = self.draft
        self.draft = None
        self._needs_redraw = True

    def _delete(self):
        if not self.rois:
            print("[delete] no ROIs saved yet")
            return
        print("[delete] existing ROIs:")
        for name in sorted(self.rois.keys()):
            print(f"  - {name}")
        name = self._prompt("Name to delete (blank to cancel, '*' to clear all): ")
        if not name:
            print("[delete] cancelled")
            return
        if name == "*":
            answer = self._prompt(f"Delete ALL {len(self.rois)} ROIs? [y/N] ")
            if answer.lower() in ("y", "yes"):
                self.rois.clear()
                save_rois(self.image_path, self.rois)
                print("[delete] all ROIs removed")
                self._needs_redraw = True
            return
        if name not in self.rois:
            print(f"[delete] no ROI named '{name}'")
            return
        self.rois.pop(name)
        save_rois(self.image_path, self.rois)
        print(f"[delete] removed '{name}'")
        self._needs_redraw = True

    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)
        print(f"editing ROIs for {self.image_path.name}")
        print(f"saving to {rois_path_for(self.image_path)}")
        print("controls: drag mouse = draw, s = save draft, d = delete, q/ESC = quit")

        while True:
            if self._needs_redraw:
                self._render()
                self._needs_redraw = False

            key = cv2.waitKey(20) & 0xFF
            if key == 0xFF:
                continue
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                self._save_draft()
            elif key == ord("d"):
                self._delete()
            elif key == ord("c"):
                if self.draft is not None:
                    self.draft = None
                    self._needs_redraw = True
                    print("[draft] cleared")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--image",
        default=None,
        help="Image filename inside aria_ispy/assets/ (default: source from tv_layout.json or first asset).",
    )
    args = parser.parse_args()

    image_path = resolve_image(args.image)
    editor = ROIEditor(image_path)
    editor.run()


if __name__ == "__main__":
    main()
