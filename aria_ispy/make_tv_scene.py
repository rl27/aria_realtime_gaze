import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
RUNTIME = ROOT / "runtime"

TV_W = 1920
TV_H = 1080

TAG_SIZE = 250
MARGIN = 24
TAG_GAP = 16

ISPY_PATH = ASSETS / "ispy.jpg"
TAG_ID_TO_POSITION = {
    0: "top_left",
    1: "top_right",
    2: "bottom_left",
    3: "bottom_right",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", default=str(ISPY_PATH))
    p.add_argument("--out-dir", default=str(RUNTIME))
    p.add_argument("--tag-size", type=int, default=TAG_SIZE)
    p.add_argument("--tag-gap", type=int, default=TAG_GAP)
    return p.parse_args()


def resolve_image_path(path):
    image_path = Path(path)
    if image_path.exists():
        return image_path

    if image_path == ISPY_PATH:
        candidates = sorted(
            p
            for p in ASSETS.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if candidates:
            print(f"Default assets/ispy.jpg not found; using {candidates[0]}")
            return candidates[0]

    raise FileNotFoundError(
        f"Missing {image_path}. Put I Spy images in aria_ispy/assets/ "
        "or pass one explicitly, e.g. --image assets/ispy1.jpg."
    )


def main():
    args = parse_args()
    image_path = resolve_image_path(args.image)
    runtime = Path(args.out_dir)
    runtime.mkdir(parents=True, exist_ok=True)

    canvas = np.full((TV_H, TV_W, 3), 255, dtype=np.uint8)

    # Maximize the displayed image while keeping it clear of the corner tags.
    image_rect_tv = [
        MARGIN + TAG_SIZE + args.tag_gap,
        MARGIN,
        TV_W - MARGIN - args.tag_size - args.tag_gap,
        TV_H - MARGIN,
    ]

    ispy = Image.open(image_path).convert("RGB")
    orig_w, orig_h = ispy.size
    x0, y0, x1, y1 = image_rect_tv
    max_w = x1 - x0
    max_h = y1 - y0

    resample = getattr(Image, "Resampling", Image).LANCZOS
    scale = min(max_w / orig_w, max_h / orig_h)
    display_w = int(round(orig_w * scale))
    display_h = int(round(orig_h * scale))
    ispy_display = ispy.resize((display_w, display_h), resample)
    ispy_np = np.array(ispy_display)

    img_h, img_w = ispy_np.shape[:2]

    img_x0 = x0 + (max_w - img_w) // 2
    img_y0 = y0 + (max_h - img_h) // 2
    img_x1 = img_x0 + img_w
    img_y1 = img_y0 + img_h

    canvas[img_y0:img_y1, img_x0:img_x1] = ispy_np

    cv2.rectangle(
        canvas,
        (img_x0, img_y0),
        (img_x1, img_y1),
        (0, 0, 0),
        2,
    )

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)

    # Required convention:
    # ID 0 = top-left, ID 1 = top-right, ID 2 = bottom-left, ID 3 = bottom-right.
    tag_positions = {
        0: (MARGIN, MARGIN),
        1: (TV_W - MARGIN - args.tag_size, MARGIN),
        2: (MARGIN, TV_H - MARGIN - args.tag_size),
        3: (TV_W - MARGIN - args.tag_size, TV_H - MARGIN - args.tag_size),
    }

    tag_corners_tv = {}

    for tag_id, (tx, ty) in tag_positions.items():
        marker = cv2.aruco.generateImageMarker(
            dictionary,
            tag_id,
            args.tag_size,
        )

        marker_rgb = cv2.cvtColor(marker, cv2.COLOR_GRAY2RGB)
        canvas[ty : ty + args.tag_size, tx : tx + args.tag_size] = marker_rgb

        tag_corners_tv[str(tag_id)] = [
            [tx, ty],
            [tx + args.tag_size, ty],
            [tx + args.tag_size, ty + args.tag_size],
            [tx, ty + args.tag_size],
        ]

    layout = {
        "tv_width": TV_W,
        "tv_height": TV_H,
        "tag_family": "DICT_APRILTAG_36h11",
        "tag_size_px": args.tag_size,
        "tag_id_to_position": {str(k): v for k, v in TAG_ID_TO_POSITION.items()},
        "tag_corners_tv": tag_corners_tv,
        "image_rect_tv": [img_x0, img_y0, img_x1, img_y1],
        "ispy_image_size_px": [orig_w, orig_h],
        "displayed_image_size_tv_px": [img_w, img_h],
        "source_image_name": image_path.name,
        "source_image_path": str(image_path),
        "tag_gap_px": args.tag_gap,
    }

    cv2.imwrite(
        str(runtime / "tv_scene.png"),
        cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
    )

    with open(runtime / "tv_layout.json", "w") as f:
        json.dump(layout, f, indent=2)

    print("Wrote runtime/tv_scene.png")
    print("Wrote runtime/tv_layout.json")
    print("Display runtime/tv_scene.png fullscreen on the TV.")


if __name__ == "__main__":
    main()
