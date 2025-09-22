#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Keep these three lines at the top. They prevent more than one instace to run.
from single_instance import SingleInstance
lock = SingleInstance()
lock.acquire()
# -----------------------------------------------------------------------------

import os
import cv2
import sys
import time
import pathlib
import numpy as np
import open3d as o3d
from utils import Colors, FileHelper
from focus_assistant import FocusAssistant
from pypylon import pylon

from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QSlider
from PyQt5.QtWidgets import QOpenGLWidget
from PyQt5.QtWidgets import QMainWindow, QAction, QMenuBar
from OpenGL.GL import * # type: ignore


# -----------------------------------------------------------------------------
# OpenGL video widget
# -----------------------------------------------------------------------------
class GLVideoWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = None # numpy uint8 HxWx3 RGB
        self.texture_id = None
        self._w = 1
        self._h = 1
        self._frame_w = 1280   # default before first frame
        self._frame_h = 1024
        self.board_corners_history = []   # store last N borders
        self.max_history = 20

        self.mesh_vertices = None
        self.mesh_triangles = None
        self.mesh_normals = None

    def load_stl(self, filepath):
        verts, tris, norms = load_stl_with_open3d(filepath)
        self.mesh_vertices = verts
        self.mesh_triangles = tris
        self.mesh_normals = norms
        self.update()

    def set_board_corners(self, pts):
        self.board_corners_history.append(np.asarray(pts, dtype=np.float32))
        if len(self.board_corners_history) > self.max_history:
            self.board_corners_history.pop(0)
        self.update()

    def sizeHint(self):
        return QSize(self._frame_w, self._frame_h)

    def set_frame(self, rgb_frame: np.ndarray):
        self.frame = rgb_frame
        h, w, _ = rgb_frame.shape
        if (w, h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = w, h
            self.updateGeometry()   # notify layout that sizeHint changed
        self.update()

    def initializeGL(self):
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)

    def resizeGL(self, w, h):
        self._w = max(1, w)
        self._h = max(1, h)
        glViewport(0, 0, self._w, self._h)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT)
        if self.frame is None:
            return

        h, w, _ = self.frame.shape

        # Upload texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, self.frame)

        # Draw textured quad filling the viewport
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0,  1.0)
        glEnd()

        # --- Draw STL mesh ---
        if self.mesh_vertices is not None:
            print ('HERE')
            self._draw_stl()
        
        # --- Draw CalibPlate's frame ---
        if self.board_corners_history:
            glDisable(GL_TEXTURE_2D)
            glLineWidth(2.0)

            n = len(self.board_corners_history)
            for i, corners in enumerate(self.board_corners_history):
                alpha = (i + 1) / n    # older = fainter
                glColor4f(1.0, 0.0, 0.0, alpha)  # RGBA, requires blending

                glBegin(GL_LINE_LOOP)
                for (px, py) in corners:
                    ndc_x = (px / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (py / h) * 2.0
                    glVertex2f(ndc_x, ndc_y)
                glEnd()

            glEnable(GL_TEXTURE_2D)


    def _draw_stl(self):
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

        glColor3f(0.7, 0.7, 0.9)  # bluish gray

        glPushMatrix()
        glScalef(0.01, 0.01, 0.01)   # scale down if large
        glTranslatef(0, 0, -5)       # move into view

        glBegin(GL_TRIANGLES)
        for tri, n in zip(self.mesh_triangles, self.mesh_normals): # type: ignore
            glNormal3fv(n)
            for idx in tri:
                glVertex3fv(self.mesh_vertices[idx]) # type: ignore
        glEnd()

        glPopMatrix()

        glEnable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)

# -----------------------------------------------------------------------------
# Camera discovery functions
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Video capture functions
# -----------------------------------------------------------------------------
    print(os.getcwd())
    mesh = o3d.io.read_triangle_mesh(str(pathlib.Path(__file__).parent / "base_link.stl"))
    print(mesh)

def grab_and_show(ip: str) -> None:
    """ Open a Basler camera by IP and stream video until ESC is pressed."""
    di = pylon.DeviceInfo()
    di.SetIpAddress(ip)

    cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))

    try:
        cam.Open()
        cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        converter = pylon.ImageFormatConverter()
        converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        print(f"Streaming from camera at {ip}. Press ESC to exit.")

        while cam.IsGrabbing():
            res = cam.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            try:
                if res.GrabSucceeded():
                    rgb = converter.Convert(res)
                    img_array = rgb.GetArray()

                    # Convert RGB → BGR for OpenCV
                    bgr_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

                    # Show live video
                    cv2.imshow("Basler Live Video", bgr_array)

                    # Exit on ESC
                    if cv2.waitKey(1) == 27:
                        break
                else:
                    raise RuntimeError(
                        f"Grab failed: {res.ErrorCode} {res.ErrorDescription}"
                    )
            finally:
                res.Release()

    finally:
        cam.Close()
        cv2.destroyAllWindows()


# -----------------------------------------------------------------------------
# Video capture GUI
# -----------------------------------------------------------------------------
class VideoApp(QMainWindow):
    def __init__(self, ip: str, calib_path: str):
        super().__init__()
        self.setWindowTitle("Calian Robotics")
        self.focus_assistant = FocusAssistant()

        # Camera setup
        self.ip = ip
        self.cam = None
        self.converter = None
        self.focus_active = True
        self._plate_seen_start = None  # timestamp of when plate was first detected

        # Environment setup
        self.calib_path = calib_path

        self.prev_time = None
        self.fps = 0.0

        # Grid state variables
        self.rows = None
        self.cols = None
        self.cell_h = None
        self.cell_w = None
        self.cell_active = np.zeros((0, 0), dtype=bool)

        # Widgets
        self.gl = GLVideoWidget(self)
        self.label = QLabel("Camera feed will appear here")
        self.start_button = QPushButton("Start")
        self.focus_button = QPushButton("Focus")
        self.clear_button = QPushButton("Clear")
        self.stop_button = QPushButton("Stop")
        self.exposure_slider = QSlider(Qt.Horizontal) # type: ignore
        self.exposure_slider.setEnabled(False)  # enabled once camera is open

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        # layout.addWidget(self.label)
        layout.addWidget(self.gl, alignment=Qt.AlignCenter)  # type: ignore
        layout.addWidget(self.start_button)
        layout.addWidget(self.focus_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(QLabel("Exposure (µs):"))
        layout.addWidget(self.exposure_slider)
        self.setCentralWidget(central_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File") # type: ignore
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close) # type: ignore
        file_menu.addAction(exit_action) # type: ignore

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Options
        self.axes_colors=(  (0, 255, 255),  # X = yellow 
                          (255, 0, 255),  # Y = magenta
                          (255, 255, 0))  # Z = cyan

        # Button actions
        self.start_button.clicked.connect(self.start_video)
        self.focus_button.clicked.connect(self.focus_assist)
        self.clear_button.clicked.connect(self.clear_images)
        self.stop_button.clicked.connect(self.stop_video)
        self.exposure_slider.valueChanged.connect(self.set_exposure)

        # --- ArUco/ChArUco setup ---
        # Pick a predefined ArUco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        self.board = cv2.aruco.CharucoBoard(
            (7, 7),        # squaresX, squaresY
            0.020,          # squareLength (meters)
            0.016,          # markerLength (meters)
            self.dictionary # dictionary
        )

        self.squaresX, self.squaresY = 7, 7
        self.square_len = 0.020  # meters

        # Example intrinsics (must be replaced with your calibration!)
        fx = 800  # focal length in pixels (guess)
        fy = 800
        cx = 640  # image center x (assuming 1280x720 image)
        cy = 512  # image center y

        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0,  0,  1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)  # assume no distortion

        self.start_video()

    def draw_arrow(self, img, start, end, color, thickness=2, arrow_magnitude=20, angle=30):
        """
        Draw a custom arrow with adjustable tip angle.
        :param img: image to draw on
        :param start: (x, y) start point
        :param end: (x, y) end point
        :param color: BGR tuple
        :param thickness: line thickness
        :param arrow_magnitude: length of the arrowhead lines in pixels
        :param angle: half angle of the arrowhead in degrees
        """
        # Ensure plain Python int tuples (robust to NumPy arrays/scalars)
        start = tuple(map(int, np.array(start).flatten()[:2]))
        end   = tuple(map(int, np.array(end).flatten()[:2]))
        color = tuple(int(c) for c in color)

        # Draw shaft
        try:
            cv2.line(img, start, end, color, thickness)

            dx, dy = start[0] - end[0], start[1] - end[1]
            length = np.hypot(dx, dy)
            if length == 0:
                return

            ux, uy = dx / length, dy / length

            # Arrowhead
            left_x = end[0] + arrow_magnitude * (ux * np.cos(np.radians(angle)) - uy * np.sin(np.radians(angle)))
            left_y = end[1] + arrow_magnitude * (ux * np.sin(np.radians(angle)) + uy * np.cos(np.radians(angle)))

            right_x = end[0] + arrow_magnitude * (ux * np.cos(-np.radians(angle)) - uy * np.sin(-np.radians(angle)))
            right_y = end[1] + arrow_magnitude * (ux * np.sin(-np.radians(angle)) + uy * np.cos(-np.radians(angle)))

            cv2.line(img, end, (int(left_x), int(left_y)), color, thickness)
            cv2.line(img, end, (int(right_x), int(right_y)), color, thickness)
        finally:
            pass
    
    def draw_custom_axes(self, img, rvec, tvec, axis_len, colors, center=False) -> None:
        if center:
            cx = (self.squaresX * self.square_len) / 2.0
            cy = (self.squaresY * self.square_len) / 2.0
            origin = np.float32([[cx, cy, 0]]) # type: ignore
        else:
            origin = np.float32([[0, 0, 0]]) # type: ignore

        axis_pts = np.float32([
            origin[0], # type: ignore
            origin[0] + [axis_len, 0, 0], # type: ignore
            origin[0] + [0, axis_len, 0], # type: ignore
            origin[0] + [0, 0, axis_len], # type: ignore
        ]) # type: ignore

        imgpts, _ = cv2.projectPoints(axis_pts, rvec, tvec, self.camera_matrix, self.dist_coeffs) # type: ignore
        if imgpts is None or imgpts.size < 8:
            return

        # Convert to plain Python int tuples
        pts = [(int(round(float(x))), int(round(float(y)))) for x, y in imgpts.reshape(-1, 2)]
        if len(pts) < 4:
            return

        o, x, y, z = pts
        self.draw_arrow(img, o, x, colors[0], thickness=3, arrow_magnitude=15, angle=25)
        self.draw_arrow(img, o, y, colors[1], thickness=3, arrow_magnitude=15, angle=25)
        self.draw_arrow(img, o, z, colors[2], thickness=3, arrow_magnitude=15, angle=25)
   
    def draw_grid_circles(self, img, target_cells=24):
        h, w, _ = img.shape

        # Initialize grid layout once
        if self.rows is None:
            best_diff = float("inf")
            best_rows, best_cols = 1, target_cells
            for rows in range(1, target_cells + 1):
                cols = int(round(target_cells / rows))
                if rows * cols != target_cells:
                    continue
                cell_h = h / rows
                cell_w = w / cols
                diff = abs(cell_h - cell_w)
                if diff < best_diff:
                    best_diff = diff
                    best_rows, best_cols = rows, cols

            self.rows, self.cols = best_rows, best_cols
            self.cell_h = h // self.rows
            self.cell_w = w // self.cols
            self.cell_active = np.zeros((self.rows, self.cols), dtype=bool)

        # Draw circles
        for r in range(self.rows):
            for c in range(self.cols): # type: ignore
                cx = int(c * self.cell_w + self.cell_w / 2) # type: ignore
                cy = int(r * self.cell_h + self.cell_h / 2) # type: ignore
                radius = int(0.75 * self.cell_w / 2) # type: ignore

                if self.cell_active[r, c]: # type: ignore
                    color = (0, 255, 0)  # solid green
                    cv2.circle(img, (cx, cy), radius, color, -1)
                else:
                    color = (255, 255, 255)
                    cv2.circle(img, (cx, cy), radius, color, 2)

        return self.rows, self.cols, self.cell_h, self.cell_w

    def activate_cell_at_point(self, imgpt):
        """
        Try to activate the grid cell containing the given point.

        Args:
            imgpt: (x, y) pixel coordinates

        Returns:
            (success: bool, cell_id: tuple[int, int] | None)
            success = True if a *new* cell was activated
            False if no cell was found or it was already active
            cell_id = (row, col) of activated cell, or (-1, -1) if none
        """
        if self.rows is None or self.cols is None:
            return False, (-1, -1)

        x, y = int(imgpt[0]), int(imgpt[1])

        for r in range(self.rows):
            for c in range(self.cols):
                cx = int(c * self.cell_w + self.cell_w / 2) # type: ignore
                cy = int(r * self.cell_h + self.cell_h / 2) # type: ignore
                radius = int(0.75 * self.cell_w / 2) # type: ignore

                # Distance from circle center
                dist = np.hypot(x - cx, y - cy)

                if dist <= radius:
                    # Already active? → return False
                    if self.cell_active[r, c]:
                        return False, (r, c)

                    # Newly activated
                    self.cell_active[r, c] = True
                    return True, (r, c)

        return False, (-1, -1)

    def keyPressEvent(self, event): # type: ignore
        """Exit cleanly when ESC is pressed."""
        if event.key() == Qt.Key_Escape: # type: ignore
            self.close()

    def start_video(self):
        if self.cam is None:
            di = pylon.DeviceInfo()
            di.SetIpAddress(self.ip)
            self.cam = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(di))
            self.cam.Open()

            # Configure exposure slider range
            exp_min = int(self.cam.ExposureTimeAbs.GetMin())
            exp_max = int(self.cam.ExposureTimeAbs.GetMax())
            exp_current = int(self.cam.ExposureTimeAbs.GetValue())
            self.exposure_slider.setRange(exp_min, exp_max)
            self.exposure_slider.setValue(exp_current)
            self.exposure_slider.setEnabled(True)

        # (Re)create converter here every time
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_RGB8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        if not self.cam.IsGrabbing():
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.timer.start(30)

    def focus_assist(self):
        self.focus_active = not self.focus_active

    def clear_images(self):
        pass

    def stop_video(self):
        self.timer.stop()
        if self.cam:
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
            # Don’t close/release here if you want to restart easily
        self.label.setText("Camera stopped")

    def set_exposure(self, value):
        if self.cam and self.cam.IsOpen():
            try:
                self.cam.ExposureAuto.SetValue("Off")  # disable auto
                self.cam.ExposureTimeAbs.SetValue(value)
            except Exception as e:
                print("Exposure set failed:", e)

    def update_frame(self):
        res = None
        
        if not (self.cam and self.cam.IsGrabbing() and self.converter):
            return
        
        try:
            res = self.cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)    # raise exception if no new frame arrives whithin that timeout
            if res.GrabSucceeded():
                rgb = self.converter.Convert(res)
                img_array = rgb.GetArray()
                h, w, _ = img_array.shape
                border_color = None

                # ---- Make overlay copy ----
                overlay = img_array.copy()

                # ---- Draw transparent grid circles (background) ----
                grid = overlay.copy()
                if not self.focus_active:
                    self.draw_grid_circles(grid, 24)
                    cv2.addWeighted(grid, 0.3, overlay, 0.7, 0, overlay)

                # ---- FPS calculation ----
                now = time.time()
                if self.prev_time is not None:
                    dt = now - self.prev_time
                    if dt > 0:
                        self.fps = 1.0 / dt
                self.prev_time = now

                # ---- Convert to grayscale for detection (clean image, no drawings) ----
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # --- Detect ArUco markers ---
                corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

                new_corners_added = False

                if ids is not None and len(ids) > 0:
                    # Draw detected markers on overlay
                    cv2.aruco.drawDetectedMarkers(overlay, corners, ids)

                    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                        markerCorners=corners,
                        markerIds=ids,
                        image=gray,
                        board=self.board
                    )

                    if retval is not None and retval >= 6:
                        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                            charuco_corners, charuco_ids, self.board,
                            self.camera_matrix, self.dist_coeffs, None, None # type: ignore
                        ) # type: ignore

                        pose_drawn = False

                        if ok and rvec is not None and tvec is not None:
                            # 3D board corners in board coordinates (Z=0 plane)
                            L = self.square_len
                            sX, sY = self.squaresX, self.squaresY
                            board_corners_obj = np.float32([
                                [0,    0,    0],
                                [sX*L, 0,    0],
                                [sX*L, sY*L, 0],
                                [0,    sY*L, 0],
                            ]) # type: ignore
                            imgpts, _ = cv2.projectPoints(board_corners_obj, rvec, tvec, self.camera_matrix, self.dist_coeffs) # type: ignore
                            self.gl.set_board_corners(imgpts.reshape(-1, 2))
                            new_corners_added = True

                            # --- Project the board origin ---
                            origin_imgpt, _ = cv2.projectPoints(
                                np.array([[0, 0, 0]], dtype=np.float32),
                                rvec, tvec,
                                self.camera_matrix, self.dist_coeffs
                            )
                            origin_imgpt = origin_imgpt.reshape(2)

                            # Green debug dot at origin
                            cv2.circle(overlay, (int(origin_imgpt[0]), int(origin_imgpt[1])), 12, (0, 255, 0), -1)

                            # -------------------------------------------------
                            # Cell activation logic/actions
                            if not self.focus_active:
                                success, cell_id = self.activate_cell_at_point(origin_imgpt)    # Activate circle if origin is inside one
                                if success:
                                    print(f"{Colors().text(str(cell_id), 'green')}")
                                    
                                    # Capture image and save the image
                                    r, c = cell_id
                                    filename = f"cell_{r}_{c}.png"
                                    filepath = os.path.join(self.calib_path, filename)

                                    # Save the current overlay image (with drawings) to file
                                    cv2.imwrite(filepath, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
                                    print(f"Saved raw image: {filepath}")
                                # -------------------------------------------------

                            # Save copy before drawing axes
                            before_axes = overlay.copy()
                            self.draw_custom_axes(overlay, rvec, tvec, axis_len=0.05, colors=self.axes_colors, center=True)

                            # Check if axes changed the overlay
                            if not np.array_equal(before_axes, overlay):
                                pose_drawn = True
                                border_color = (0, 255, 0)  # green
                        
                        else:
                            # Lost detection: reset timer
                            self._plate_seen_start = None

                        if not pose_drawn:
                            border_color = (255, 0, 0)  # red (pose unstable or axes skipped)

                    elif retval is not None and retval > 0:
                        border_color = (255, 255, 0)  # yellow (corners detected but too few for pose)

                # ---- Draw border if needed ----
                if border_color is not None:
                    thickness = 16
                    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), border_color, thickness)

                # ---- FPS text ----
                bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    bgr,
                    f"FPS: {self.fps:.1f}",
                    (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # --- If nothing added, decay history ---
                if not new_corners_added and self.gl.board_corners_history:
                    self.gl.board_corners_history.pop(0)
                    self.gl.update()

                # ---- Convert back to RGB for Qt ----
                # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                # h, w, ch = rgb.shape
                # bytes_per_line = ch * w
                # qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
                # self.label.setPixmap(QPixmap.fromImage(qimg))

                # --- Focus assistant ---
                if self.focus_active:
                    norm, hint, improving = self.focus_assistant.get_focus_info(img_array)
                    self.focus_assistant.draw_focus_bar(bgr, norm, hint, improving)  # draw on bgr, not overlay

                # Convert back to RGB for OpenGL upload
                rgb_for_gl = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                self.gl.set_frame(rgb_for_gl)
        
        except pylon.TimeoutException:
            print(Colors.text("Connection with camera lost! (timeout)", "red"))

        # except pylon.RuntimeException as e:
        #     print(colors.text(f"Pylon error: {e}", "yellow"))
        
        finally:
            # Only release the grab result if we actually got one
            if res is not None:
                res.Release()

    def closeEvent(self, event): # type: ignore
        """Ensure camera closes cleanly when window exits."""
        self.stop_video()
        if self.cam:
            self.cam.Close()
            self.cam = None
        event.accept()


# -----------------------------------------------------------------------------
# STL
# -----------------------------------------------------------------------------
def load_stl_with_open3d(filepath):
    mesh = o3d.io.read_triangle_mesh(filepath)
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    triangles = np.asarray(mesh.triangles, dtype=np.int32)
    normals = np.asarray(mesh.triangle_normals, dtype=np.float32)

    return vertices, triangles, normals



# -----------------------------------------------------------------------------
# Main entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    devices = list_cameras()

    device_name = "Edgar"
    ip = get_camera_ip_by_id(device_name)
    calib_path = f"calibData_{device_name}"
    
    print(f"IP: {ip}")
    print(f"calibData Path: {calib_path}")

    window = VideoApp(ip, calib_path=calib_path)    
    file_helper = FileHelper()
    file_helper.create_directory(calib_path, force_replace=True)

    path_to_stl = str(pathlib.Path(__file__).parent / "base_link.stl")
    # window.gl.load_stl(path_to_stl)
    window.show()
    sys.exit(app.exec_())

# %%
