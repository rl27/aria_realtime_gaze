# Maze game with Aria glasses and eye gaze estimation

Setup:
* Python version 3.11.12
* `pip install -r requirements.txt`

## Streaming instructions

USB:
1. When connected via USB: `python rgb_eye_dot.py`
2. Do not disconnect USB

Glasses hotspot: 
1. While connected via USB: `aria streaming start --interface hotspot --use-ephemeral-certs`
2. Wait for glasses hotspot to start, then connect computer to hotspot using password shown in terminal
3. `python rgb_eye_dot.py --interface subscribe`
4. Can disconnect USB now

Wifi:
1. Turn on phone hotspot (university wifi doesn't work, haven't tested with other wifi)
2. Connect glasses to hotspot using Aria companion app, connect computer to hotspot
2. `python rgb_eye_dot.py --interface wifi --device-ip [glasses ip listed in companion app]`

## Streaming instructions

Press 'c' to calibrate gaze.

Press 't' to reset the position to the start of the maze and reset the ending celebration.

Press 'r' to randomize the maze, reset the position, and reset the celebration.

## References

Reference: https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/samples/streaming_subscribe

Setup references:
* https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup
* https://github.com/facebookresearch/projectaria_eyetracking (this is also the model source)
