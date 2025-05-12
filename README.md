# Real-time gaze estimation and object detection with Aria glasses

Setup:
* Python verison 3.11.12
* `pip install -r requirements.txt`

All versions of `rgb_eye` default to using USB connection and Metal Performance Shaders.
* `python rgb_eye.py` runs object detection on the eye gaze region with our method of reducing inferences. Also saves a video recording of the outputs.
* `python rgb_eye_full.py` runs object detection on the eye gaze region on every frame.
* `python rgb_eye_dot.py` visualizes the eye gaze as a single dot.
* `python device_stream.py` visualizes all sensors in the glasses.

Setup references:
* https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup
* https://github.com/facebookresearch/projectaria_eyetracking (this is also the model source)