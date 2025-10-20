import os
import cv2 
import time
import pathlib
import numpy as np

from pypylon import pylon
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow,
    QMessageBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSlider,
    QAction,
    QSizePolicy,
)
from camera_utils import *
from gl_video_widget import GLVideoWidget
from focus_assistant import FocusAssistant
from pose_smoother import PoseSmoother
from utils import Colors, FileHelper


class VideoApp(QMainWindow):
    def __init__(self, ip: str, stl_path: str | None = None):
        super().__init__()

        self.setWindowTitle("Calian Robotics")
        self.focus_assistant = FocusAssistant()
        self.pose_smoother = PoseSmoother(alpha=0.9, timeout=0.2)

        # Camera setup
        self.ip = ip
        self.cam = None
        self.converter = None
        self.focus_active = True
        self._plate_seen_start = None  # timestamp of when plate was first detected

        # Default calibration directory (updated when 'Calibrate' is pressed)
        script_dir = pathlib.Path(__file__).resolve().parent
        base_dir = script_dir / "calibData"
        base_dir.mkdir(exist_ok=True)
        self.calib_path = str(base_dir)

        # Environment setup
        self.display_scale = 1.0

        self.prev_time = None
        self.fps = 0.0
        self._feed_w: int | None = None
        self._feed_h: int | None = None

        # overlays
        self.stability_progress = 0.0

        # Grid state variables
        self.rows = None
        self.cols = None
        self.cell_h = None
        self.cell_w = None
        self.cell_active = np.zeros((0, 0), dtype=bool)
        self.current_r = -1
        self.current_c = -1

        # Widgets
        self.gl = GLVideoWidget(self)
        self.gl.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        if stl_path:
            self.gl.load_stl(stl_path)

        self.label = QLabel("Camera feed will appear here")
        self.label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.focus_button = QPushButton("Focus")
        self.clear_button = QPushButton("Clear")
        self.calibrate_button = QPushButton("Calibrate")

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        # layout.addWidget(self.label)
        layout.addWidget(self.gl, alignment=Qt.AlignCenter)  # type: ignore
        layout.addWidget(self.focus_button)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.calibrate_button)
        self.setCentralWidget(central_widget)

        # Menu bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File") # type: ignore
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close) # type: ignore
        file_menu.addAction(exit_action) # type: ignore

        view_menu = menubar.addMenu("View") # type: ignore
        self.action_toggle_border = QAction("Use Image-Aligned Border", self, checkable=True, checked=False) # type: ignore
        self.action_toggle_border.toggled.connect(self.on_toggle_border)
        view_menu.addAction(self.action_toggle_border) # type: ignore

        # Camera switching
        raw_cameras = list_cameras()
        self.available_cameras = {}

        # Handle tuple or list of DeviceInfo objects
        if isinstance(raw_cameras, (list, tuple)):
            for dev_info in raw_cameras:
                try:
                    name = dev_info.GetFriendlyName()
                    self.available_cameras[name] = dev_info
                except Exception:
                    pass
        elif isinstance(raw_cameras, dict):
            self.available_cameras = raw_cameras

        if not self.available_cameras:
            raise RuntimeError("No cameras found.")

        # Pick the first camera by name
        self.current_camera_name = next(iter(self.available_cameras.keys()))
        device_name = self.current_camera_name

        # Camera menu
        camera_menu = menubar.addMenu("Camera")
        for cam_name in self.available_cameras.keys():
            act = QAction(cam_name, self, checkable=True, checked=(cam_name == device_name))
            act.triggered.connect(lambda checked, name=cam_name: self.switch_camera(name))
            camera_menu.addAction(act)
        self.camera_actions = camera_menu.actions()

        # Button actions
        self.focus_button.clicked.connect(self.toggle_focus)
        self.clear_button.clicked.connect(self.clear_images)
        self.calibrate_button.clicked.connect(self.start_calibration)

        # Timer for video updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Options
        self.axes_colors=(  (0, 255, 255),  # X = yellow 
                          (255, 0, 255),  # Y = magenta
                          (255, 255, 0))  # Z = cyan

        # --- ArUco/ChArUco setup ---
        # Pick a predefined ArUco dictionary
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)

        self.board = cv2.aruco.CharucoBoard(
            (7, 7),         # squaresX, squaresY
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

        self.gl.configure_board(self.squaresX, self.squaresY, self.square_len)
        self.gl.set_camera_calibration(self.camera_matrix, self.dist_coeffs)

        self.start_video()

    def ensure_feed_size(self, width: int, height: int) -> None:
        """Keep the video display widgets matched to the camera resolution,
        with optional scaling for display."""
        if width <= 0 or height <= 0:
            return

        if self._feed_w == width and self._feed_h == height:
            return

        self._feed_w, self._feed_h = width, height

        # Apply display scaling only to Qt widgets
        disp_w = int(width * self.display_scale)
        disp_h = int(height * self.display_scale)

        self.gl.setFixedSize(disp_w, disp_h)
        self.gl.updateGeometry()
        self.label.setFixedSize(disp_w, disp_h)

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
            origin[0] + [0, 0, -axis_len], # type: ignore
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
                    if self.current_r == r and self.current_c == c:
                        color = (255, 165, 0)
                        cv2.circle(img, (cx, cy), radius, color, -1)
                    else:
                        color = (255, 255, 255)
                        cv2.circle(img, (cx, cy), radius, color, 2)

        return self.rows, self.cols, self.cell_h, self.cell_w

    def draw_progress_ring(self, img, center, radius, percentage, color=(0, 255, 0), thickness=4):
        """
        Draw a ring (arc) around a point showing completion percentage (0–1).
        """
        if percentage <= 0:
            return
        percentage = np.clip(percentage, 0.0, 1.0)

        # Arc angle span (0–360)
        angle_end = int(360 * percentage)

        # Draw arc
        axes = (radius + 5, radius + 5)
        cv2.ellipse(
            img,
            center,
            axes,
            0,              # rotation
            0,              # startAngle
            angle_end,      # endAngle
            color,
            thickness,
            cv2.LINE_AA,
        )

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
    
    def is_at_cell(self, imgpt):
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
                        return False, (-1, -1)

                    return True, (r, c)

        return False, (-1, -1)

    def max_overlap(self, corners_history: list[np.ndarray], tol: float = 1.0) -> tuple[bool, int]:
        """
        Return (stable, max_count):
        - stable = True if all sets overlap within tolerance
        - max_count = maximum overlap count observed
        """
        if not corners_history:
            return False, 0

        # Normalize history: (4,2) arrays, sorted
        sets = []
        for corners in corners_history:
            c = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            sets.append(np.array(sorted(c.tolist())))

        n = len(sets)
        max_count = 1

        for i in range(n):
            count = 1
            for j in range(i + 1, n):
                # compute per-corner Euclidean distances
                dists = np.linalg.norm(sets[i] - sets[j], axis=1)
                avg_dist = float(np.mean(dists))
                if avg_dist < tol:
                    count += 1
            max_count = max(max_count, count)

        return len(corners_history) == max_count, max_count

    def keyPressEvent(self, event): # type: ignore
        """Exit cleanly when ESC is pressed."""
        if event.key() == Qt.Key_Escape: # type: ignore
            self.close()

    def on_toggle_border(self, checked: bool):
        self.gl.set_overlay_mode("image" if checked else "ideal")

    def toggle_focus(self):
        self.focus_active = not self.focus_active

    def clear_images(self):
        """Clear any activated grid cells and refresh the view."""
        # If the grid has been initialized, reset activation state
        if isinstance(self.cell_active, np.ndarray) and self.cell_active.size > 0:
            self.cell_active[...] = False
        # Optional: also clear the fading red board-corner trails in the GL overlay
        # Comment this out if you want to keep the trails when clearing cells
        if hasattr(self.gl, "board_corners_history"):
            self.gl.board_corners_history.clear()

        # Force an immediate visual refresh
        self.gl.update()

    def start_video(self):
        if self.cam is None:
            self.cam = create_camera(self.ip)
            self.cam.Open()

        # (Re)create converter every time
        self.converter = make_converter()

        if not self.cam.IsGrabbing():
            from pypylon import pylon  # local import, only for the enum
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.timer.start(1)    
    
    def stop_video(self):
        self.timer.stop()
        if self.cam:
            if self.cam.IsGrabbing():
                self.cam.StopGrabbing()
            # Don’t close/release here if you want to restart easily
        self.label.setText("Camera stopped")

    def closeEvent(self, event): # type: ignore
        """Ensure camera closes cleanly when window exits."""
        self.stop_video()
        if self.cam:
            self.cam.Close()
            self.cam = None
        event.accept()

    def switch_camera(self, camera_name: str):
        """Switch to a different connected camera."""
        if camera_name == self.current_camera_name:
            return

        print(f"Switching camera from {self.current_camera_name} → {camera_name}")
        self.stop_video()

        try:
            # Close current
            if self.cam:
                self.cam.Close()
                self.cam = None

            # Open new
            from camera_utils import get_camera_ip_by_id
            ip = get_camera_ip_by_id(camera_name)
            self.cam = create_camera(ip)
            self.cam.Open()

            self.converter = make_converter()
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.current_camera_name = camera_name
            print(f"Camera switched to {camera_name}")

            # Update menu checks
            for act in self.camera_actions:
                act.setChecked(act.text() == camera_name)

            self.timer.start(1)

        except Exception as e:
            print(f"Failed to switch to {camera_name}: {e}")

    def start_calibration(self):
        """
        Create or reset a calibration directory for the currently selected camera.
        Warns the user before deleting any existing data.
        """
        # --- Ensure we have a valid camera name ---
        cam_name = getattr(self, "current_camera_name", None)
        if not cam_name:
            QMessageBox.warning(self, "No Camera Selected", "No active camera to calibrate.")
            return

        # Normalize folder name
        cam_name = cam_name.replace(" ", "_").replace("(", "").replace(")", "")

        # --- Build full directory path ---
        script_dir = pathlib.Path(__file__).resolve().parent
        base_dir = script_dir / "calibData"
        base_dir.mkdir(exist_ok=True)
        target_dir = base_dir / cam_name

        helper = FileHelper(parent=self)

        # --- If directory exists, ask before replacing ---
        if target_dir.exists():
            reply = QMessageBox.warning(
                self,
                "Overwrite Existing Data",
                f"A folder for '{cam_name}' already exists.\n"
                "Continuing will delete any previous calibration files.\n\n"
                "Do you want to continue?",
                QMessageBox.Yes | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if reply == QMessageBox.Cancel:
                print("Calibration canceled by user.")
                return

        # --- Update internal calibration path ---
        self.calib_path = str(target_dir)
        helper.create_directory(str(target_dir), force_replace=True)

    # ---------- main loop ----------
    def update_frame(self):
        if not (self.cam and self.cam.IsGrabbing() and self.converter):
            return
        
        res = None        
        try:
            import time
            t0 = time.time()

            res = self.cam.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)    # raise exception if no new frame arrives within that timeout
            t1 = time.time()
            if not res.GrabSucceeded():
                return
            
            img_array = self.converter.convert_and_draw(res, label="")
            t3 = time.time()

            # print(f"Grab: {(t1 - t0) * 1000:.2f} ms "
            #       f"Convert: {(t2 - t1) * 1000:.2f} ms "
            #       f"GetArray: {(t3 - t2) * 1000:.2f} ms")
            # print("Resulting FPS:", self.cam.ResultingFrameRateAbs.GetValue())
            # print("Exposure (us):", self.cam.ExposureTimeAbs.GetValue())

            pose_detected = False
            new_corners_added = False
            rvec, tvec = None, None
            origin_imgpt = None  # Ensure variable always exists

            # Ensure defaults for stability metrics
            max_count = 0
            image_stable = False

            h, w, _ = img_array.shape
            self.ensure_feed_size(w, h)

            # ---- FPS calculation ----
            now = time.time()
            if self.prev_time is not None:
                dt = now - self.prev_time
                if dt > 0:
                    self.fps = 1.0 / dt
            self.prev_time = now

            # ---- Make overlay copy ----
            overlay = img_array.copy()

            # ---- Draw transparent grid circles (background) ----
            grid = overlay.copy()
            if not self.focus_active:
                self.draw_grid_circles(grid, 24)
                cv2.addWeighted(grid, 0.5, overlay, 0.5, 0, overlay)

            # ---- Convert to grayscale for detection (clean image, no drawings) ----
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

            # --- Detect ArUco markers ---
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)

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
                    ok, est_rvec, est_tvec = cv2.aruco.estimatePoseCharucoBoard(
                        charuco_corners, charuco_ids, self.board,
                        self.camera_matrix, self.dist_coeffs, None, None # type: ignore
                    ) # type: ignore

                    if ok and est_rvec is not None and est_tvec is not None:
                        # Feed into smoother
                        self.pose_smoother.update(est_rvec, est_tvec)
                        rvec, tvec = self.pose_smoother.get_pose()
                        pose_detected = True
                # If detection failed or not enough corners, try smoother fallback
                if not pose_detected:
                    rvec, tvec = self.pose_smoother.get_pose()
                    pose_detected = rvec is not None
            else:
                # No markers at all, fallback to smoother
                rvec, tvec = self.pose_smoother.get_pose()
                pose_detected = rvec is not None

            if pose_detected and rvec is not None and tvec is not None:
                # ------- IDEAL border from model and intrinsics -------
                L = self.square_len
                sX, sY = self.squaresX, self.squaresY
                board_corners_obj = np.float32([
                    [0,    0,    0],
                    [sX*L, 0,    0],
                    [sX*L, sY*L, 0],
                    [0,    sY*L, 0],
                ]) # type: ignore
                ideal_imgpts, _ = cv2.projectPoints(board_corners_obj, rvec, tvec, self.camera_matrix, self.dist_coeffs) # type: ignore
                self.gl.push_ideal_corners(ideal_imgpts.reshape(-1, 2))
                new_corners_added = True

                # ------- IMAGE-aligned border from outer marker corners -------
                all_pts = []
                for mk in corners or []:
                    all_pts.append(mk.reshape(4, 2))
                if all_pts:
                    all_pts = np.vstack(all_pts).astype(np.float32)  # (N,2)
                    hull = cv2.convexHull(all_pts)                          # (H,1,2)
                    approx = cv2.approxPolyDP(hull, epsilon=0.01 * cv2.arcLength(hull, True), closed=True)
                    if len(approx) == 4:
                        img_quad = approx.reshape(4, 2)
                    else:
                        rect = cv2.minAreaRect(all_pts)
                        box = cv2.boxPoints(rect)
                        img_quad = box.astype(np.float32)
                    # order clockwise
                    c = img_quad.mean(axis=0)
                    ang = np.arctan2(img_quad[:, 1] - c[1], img_quad[:, 0] - c[0])
                    order = np.argsort(ang)
                    img_quad = img_quad[order]
                    self.gl.push_image_corners(img_quad)

                # --- Project the board origin ---
                origin_imgpt, _ = cv2.projectPoints(
                    np.array([[0, 0, 0]], dtype=np.float32),
                    rvec, tvec,
                    self.camera_matrix, self.dist_coeffs
                )
                origin_imgpt = origin_imgpt.reshape(2)

                self.gl.set_board_pose(rvec, tvec)
                pose_detected = True

                # -------------------------------------------------
                # Cell activation logic/actions
                if self.gl.overlay_mode == "image":
                    image_stable, max_count = self.max_overlap(self.gl.image_corners_history, tol=1.0)
                else:
                    image_stable, max_count = self.max_overlap(self.gl.ideal_corners_history, tol=1.0)

                # Debug dot at origin
                dot_color = (0, 255, 0) if image_stable else (255, 0, 0)
                cv2.circle(overlay, (int(origin_imgpt[0]), int(origin_imgpt[1])), 12, dot_color, -1)
                
            # --- Stability visualization (progress ring) ---
            # This visualizes how close we are to stability (green dot threshold)
            history_len = len(self.gl.ideal_corners_history)
            if history_len > 0:
                completion = max_count / history_len
                completion = np.clip(completion, 0.0, 1.0)

                # Draw ring around current grid cell
                if 0 <= self.current_r < self.rows and 0 <= self.current_c < self.cols:
                    cx = int(self.current_c * self.cell_w + self.cell_w / 2)
                    cy = int(self.current_r * self.cell_h + self.cell_h / 2)
                    radius = int(0.75 * self.cell_w / 2)
                    self.draw_progress_ring(overlay, (cx, cy), radius, completion, color=(0, 255, 0), thickness=4)


                if not self.focus_active and origin_imgpt is not None:
                    if not image_stable:
                        _, cell_id = self.is_at_cell(origin_imgpt)
                        self.current_r, self.current_c = cell_id
                    else:
                        success, cell_id = self.activate_cell_at_point(origin_imgpt)
                        if success:
                            print(f"{Colors().text(str(cell_id), 'green')}")
                            # Capture and save an image sized like the on-screen feed
                            r, c = cell_id
                            filename = f"{self.current_camera_name}_cell_{r}_{c}.png"
                            
                            if not hasattr(self, "calib_path") or not self.calib_path:
                                print("No calibration path defined yet. Press 'Calibrate' first.")
                                return
                            
                            filepath = os.path.join(self.calib_path, filename)
                            save_rgb = img_array
                            target_w = self._feed_w if self._feed_w else w
                            target_h = self._feed_h if self._feed_h else h
                            if target_w != w or target_h != h:
                                interp = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
                                save_rgb = cv2.resize(save_rgb, (target_w, target_h), interpolation=interp)
                            cv2.imwrite(filepath, cv2.cvtColor(save_rgb, cv2.COLOR_RGB2BGR))

            else:
                self.gl.clear_board_pose()

            # ---- Overlay camera name above converter-drawn FPS ----
            bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            text_camera = f"{self.current_camera_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                bgr,
                text_camera,
                (20, 40),
                font,
                1.0,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # --- If nothing added, decay history ---
            if not new_corners_added and self.gl.ideal_corners_history:
                self.gl.ideal_corners_history.pop(0)
                self.gl.update()

            # --- Focus assistant ---
            if self.focus_active:
                norm, hint, improving = self.focus_assistant.get_focus_info(img_array)
                self.focus_assistant.draw_focus_bar(bgr, norm, hint, improving)

            # Convert back to RGB for OpenGL upload
            rgb_for_gl = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            self.gl.set_frame(rgb_for_gl)
        
        except pylon.TimeoutException:
            print(Colors.text("Connection with camera lost! (timeout)", "red"))
        
        finally:
            if res is not None:
                res.Release()
