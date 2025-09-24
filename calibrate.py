#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Keep these three lines at the top. They prevent more than one instace to run.
from utils import SingleInstance
lock = SingleInstance()
lock.acquire()
# -----------------------------------------------------------------------------

import os
import sys
import pathlib

from PyQt5.QtWidgets import QApplication
from utils import Colors, FileHelper
from camera_utils import *
from video_app import VideoApp

def main():
    # os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    # os.environ["QT_SCALE_FACTOR"] = "1.5"
    app = QApplication(sys.argv)
    _ = list_cameras()

    device_name = "Edgar"
    ip = get_camera_ip_by_id(device_name)
    script_dir = pathlib.Path(__file__).parent.resolve()
    calib_path = str(script_dir / f"calibData_{device_name}")
    FileHelper().create_directory(calib_path, force_replace=True)

    print(f"IP: {ip}")
    print(f"calibData Path: {calib_path}")

    # path_to_stl = str(script_dir / "resources" / "base_link.stl")
    path_to_stl = None
    window = VideoApp(ip, calib_path=calib_path, stl_path=path_to_stl)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


# %%

