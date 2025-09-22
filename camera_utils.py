from pypylon import pylon

def get_camera_ip_by_id(target_id: str) -> str:
    """Find the IP address of a camera that matches a given ID."""
    for dev in pylon.TlFactory.GetInstance().EnumerateDevices():
        if target_id in (dev.GetSerialNumber(), dev.GetFriendlyName(), dev.GetUserDefinedName()):
            if dev.GetDeviceClass() == "BaslerGigE":
                return dev.GetIpAddress()
            raise ValueError(f"Camera {target_id} found but not GigE")
    raise LookupError(f"No camera found matching {target_id}")

def list_cameras():
    """List all connected Basler cameras with their IDs."""
    factory = pylon.TlFactory.GetInstance()
    devices = factory.EnumerateDevices()

    if not devices:
        print("No cameras found.")
        return []

    print("Connected cameras:")
    for i, dev in enumerate(devices):
        print(f"[{i}] Name: {dev.GetFriendlyName()}, "
              f"Serial: {dev.GetSerialNumber()}, "
              f"IP: {dev.GetIpAddress() if dev.GetDeviceClass() == 'BaslerGigE' else 'N/A'}")
    return devices

def create_camera(ip: str) -> pylon.InstantCamera:
    """Create and return a Pylon InstantCamera for the given IP."""
    di = pylon.DeviceInfo()
    di.SetIpAddress(ip)
    return pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))


def make_converter() -> pylon.ImageFormatConverter:
    """Create and return a configured Pylon ImageFormatConverter."""
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_RGB8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter