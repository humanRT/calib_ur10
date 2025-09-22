#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Keep these three lines at the top. They prevent more than one instace to run.
from single_instance import SingleInstance
lock = SingleInstance()
lock.acquire()
# -----------------------------------------------------------------------------

import sys
import pathlib

from PyQt5.QtWidgets import QApplication

from camera_utils import *
from utils import FileHelper
from main_window import VideoApp

def main():
    app = QApplication(sys.argv)
    _ = list_cameras()

    device_name = "Edgar"
    ip = get_camera_ip_by_id(device_name)
    script_dir = pathlib.Path(__file__).parent.resolve()
    calib_path = str(script_dir / f"calibData_{device_name}")
    FileHelper().create_directory(calib_path, force_replace=True)

    print(f"IP: {ip}")
    print(f"calibData Path: {calib_path}")

    window = VideoApp(ip, calib_path=calib_path)
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()


# %%
