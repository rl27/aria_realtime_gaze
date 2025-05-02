# Real-time eye gaze with Aria glasses

Setup guide: https://facebookresearch.github.io/projectaria_tools/docs/ARK/sdk/setup

Model source: https://github.com/facebookresearch/projectaria_eyetracking
* Needed to install specific versions of torch (2.5.1) and torchvision (0.20.1)

Eye gaze visualization: `python -m rgb_eye --interface usb --update_iptables`

Visualization of all sensors: `python -m device_stream --interface usb --update_iptables`