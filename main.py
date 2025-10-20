#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Keep these three lines at the top. They prevent more than one instance from running.
from utils import SingleInstance
lock = SingleInstance()
lock.acquire()
# -----------------------------------------------------------------------------

import sys
import pathlib
from PyQt5.QtWidgets import QApplication
from utils import Colors, Screen
from camera_utils import *
from video_app import VideoApp

def main():
    Screen.clear_console()

    app = QApplication(sys.argv)
    cameras = list_cameras()

    if not cameras:
        print("No cameras detected.")
        sys.exit(1)

    # Extract names from the DeviceInfo objects
    camera_names = []
    for idx, cam_info in enumerate(cameras):
        try:
            name = cam_info.GetFriendlyName()
        except Exception:
            name = f"Camera_{idx}"
        camera_names.append(name)

    # Pick the first camera as default
    device_name = camera_names[0]
    ip = get_camera_ip_by_id(device_name)

    print(f"Using camera: {device_name}")
    print(f"IP: {ip}")

    path_to_stl = None
    window = VideoApp(ip, stl_path=path_to_stl)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()



# %%
