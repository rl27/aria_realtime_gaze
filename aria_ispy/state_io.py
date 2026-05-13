import json
import os
from pathlib import Path

import cv2


def write_player_state(state_dir, player_id, frame_bgr, state):
    """
    Atomically write:
      runtime/state/p1.json
      runtime/state/p1_frame.jpg

    The dashboard polls these files.
    """

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    frame_path = state_dir / f"{player_id}_frame.jpg"
    tmp_frame_path = state_dir / f"{player_id}_frame_tmp.jpg"
    json_path = state_dir / f"{player_id}.json"
    tmp_json_path = state_dir / f"{player_id}.json.tmp"

    cv2.imwrite(str(tmp_frame_path), frame_bgr)
    os.replace(tmp_frame_path, frame_path)

    state = dict(state)
    state["frame_path"] = str(frame_path)

    with open(tmp_json_path, "w") as f:
        json.dump(state, f, indent=2)

    os.replace(tmp_json_path, json_path)
