from pypylon import pylon
import cv2
import time


def get_camera_ip_by_id(target_id: str) -> str:
    """
    Return the IP address of a Basler GigE camera whose identifier matches
    the given target string.
    """
    target_lower = target_id.lower()
    devices = pylon.TlFactory.GetInstance().EnumerateDevices()

    if not devices:
        raise LookupError("No cameras detected by Pylon.")

    for dev in devices:
        sn = dev.GetSerialNumber().lower()
        fn = dev.GetFriendlyName().lower()
        un = dev.GetUserDefinedName().lower()

        if (target_lower in sn) or (target_lower in fn) or (target_lower in un):
            if dev.GetDeviceClass() != "BaslerGigE":
                raise ValueError(f"Camera '{target_id}' found but not a GigE device.")
            return dev.GetIpAddress()

    found = [
        f"{dev.GetFriendlyName()} ({dev.GetSerialNumber()}) [{dev.GetDeviceClass()}] @ {dev.GetIpAddress()}"
        for dev in devices
    ]
    raise LookupError(
        f"No camera found matching '{target_id}'.\n"
        f"Available devices:\n  " + "\n  ".join(found)
    )


def list_cameras(verbose=False):
    """List all connected Basler cameras with their IDs."""
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        print("No cameras found.")
        return []

    if (verbose):
        print("Connected cameras:")
        for i, dev in enumerate(devices):
            print(
                f"[{i}] Name: {dev.GetFriendlyName()}, "
                f"Serial: {dev.GetSerialNumber()}, "
                f"IP: {dev.GetIpAddress() if dev.GetDeviceClass() == 'BaslerGigE' else 'N/A'}"
            )
            
    return devices


def create_camera(ip: str) -> pylon.InstantCamera:
    """Create and return a Pylon InstantCamera for the given IP."""
    di = pylon.DeviceInfo()
    di.SetIpAddress(ip)
    return pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))


def set_exposure(cam, value):
    if cam and cam.IsOpen():
        try:
            cam.ExposureAuto.SetValue("Off")  # disable auto
            cam.ExposureTimeAbs.SetValue(value)
            print(f"Exposure set to: {value}us")
        except Exception as e:
            print("Exposure set failed:", e)


# -------------------------------------------------------------------------
# FPS overlay helper
# -------------------------------------------------------------------------
class CameraConverterWithFPS:
    """Wrapper around ImageFormatConverter that draws FPS in green at lower-left corner."""

    def __init__(self):
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        self.last_report = time.time()
        self.frame_count = 0
        self.fps = 0.0

    def convert_and_draw(self, result, label: str = ""):
        """Convert Pylon GrabResult to RGB image and draw FPS text in green (lower-left)."""
        img = self.converter.Convert(result).GetArray()
        self.frame_count += 1
        now = time.time()

        if now - self.last_report >= 1.0:
            self.fps = self.frame_count / (now - self.last_report)
            self.frame_count = 0
            self.last_report = now

        # Compute text size to position at bottom-left
        text = f"{label} {self.fps:.1f} FPS" if label else f"{self.fps:.1f} FPS"
        font_scale = 1.0
        thickness = 2
        color = (0, 255, 0)  # green
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        height, width = img.shape[:2]
        x = 10
        y = height - 10  # 10 px from bottom

        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return img


def make_converter() -> CameraConverterWithFPS:
    """Return a converter that adds FPS overlay."""
    return CameraConverterWithFPS()
