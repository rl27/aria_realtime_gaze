# Real-time gaze estimation and object detection with Aria glasses

Setup guide: https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup

Model source & further installation: https://github.com/facebookresearch/projectaria_eyetracking
* Need to install specific versions of torch (2.5.1) and torchvision (0.20.1)

All versions of `rgb_eye` default to using USB connection and Metal Performance Shaders.
* `python rgb_eye.py` runs object detection on the eye gaze region with our method of reducing inferences.
* `python rgb_eye_full.py` runs object detection on the eye gaze region on every frame.
* `python rgb_eye_dot.py` visualizes the eye gaze as a single dot.
* `python device_stream.py` visualizes all sensors in the glasses.