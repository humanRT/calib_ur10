import os
import cv2 
import time
import numpy as np

from pypylon import pylon
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow,
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
from utils import Colors


class VideoApp(QMainWindow):
    def __init__(self, ip: str, calib_path: str, stl_path : str | None = None):
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
        self._feed_w: int | None = None
        self._feed_h: int | None = None

        # overlays
        self.board_corners_history: list[np.ndarray] = []
        self.history_size = 20

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


        self.gl.configure_board(self.squaresX, self.squaresY, self.square_len)
        self.gl.set_camera_calibration(self.camera_matrix, self.dist_coeffs)

        self.start_video()

    def ensure_feed_size(self, width: int, height: int) -> None:
        """Keep the video display widgets matched to the camera resolution."""
        if width <= 0 or height <= 0:
            return

        if self._feed_w == width and self._feed_h == height:
            return

        self._feed_w, self._feed_h = width, height
        self.gl.setFixedSize(width, height)
        self.gl.updateGeometry()
        self.label.setFixedSize(width, height)

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
                    if self.current_r == r and self.current_c == c:
                        color = (255, 165, 0)
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
    
    def set_board_corners(self, pts: np.ndarray) -> None:
        self.board_corners_history.append(np.asarray(pts, dtype=np.float32))
        if len(self.board_corners_history) > self.history_size:
            self.board_corners_history.pop(0)
        self.update()

    def max_overlap(self, tol: float = 1.0) -> tuple[bool, int]:
        """
        Return the maximum number of sets of four corners that are
        within `tol` average Euclidean distance of each other.
        """
        if not self.board_corners_history:
            return 0

        # Normalize history: (4,2) arrays, sorted
        sets = []
        for corners in self.board_corners_history:
            c = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
            sets.append(np.array(sorted(c.tolist())))

        n = len(sets)
        max_count = 1

        for i in range(n):
            count = 1
            for j in range(i + 1, n):
                # compute per-corner Euclidean distances
                dists = np.linalg.norm(sets[i] - sets[j], axis=1)
                avg_dist = float(np.mean(dists))   # <-- ensure scalar
                if avg_dist < tol:
                    count += 1
            max_count = max(max_count, count)

        return self.history_size == max_count, max_count

    def keyPressEvent(self, event): # type: ignore
        """Exit cleanly when ESC is pressed."""
        if event.key() == Qt.Key_Escape: # type: ignore
            self.close()

    def start_video(self):
        if self.cam is None:
            self.cam = create_camera(self.ip)
            self.cam.Open()

            # Configure exposure slider range
            exp_min = int(self.cam.ExposureTimeAbs.GetMin())
            exp_max = int(self.cam.ExposureTimeAbs.GetMax())
            exp_current = int(self.cam.ExposureTimeAbs.GetValue())
            self.exposure_slider.setRange(exp_min, exp_max)
            self.exposure_slider.setValue(exp_current)
            self.exposure_slider.setEnabled(True)

        # (Re)create converter every time
        self.converter = make_converter()

        if not self.cam.IsGrabbing():
            from pypylon import pylon  # local import, only for the enum
            self.cam.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

        self.timer.start(30)

    def focus_assist(self):
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
                self.ensure_feed_size(w, h)
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
                pose_detected = False

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
                            self.set_board_corners(imgpts.reshape(-1, 2))
                            new_corners_added = True

                            # --- Project the board origin ---
                            origin_imgpt, _ = cv2.projectPoints(
                                np.array([[0, 0, 0]], dtype=np.float32),
                                rvec, tvec,
                                self.camera_matrix, self.dist_coeffs
                            )
                            origin_imgpt = origin_imgpt.reshape(2)

                            self.gl.set_board_pose(rvec, tvec, self.board_corners_history)
                            pose_detected = True
                            
                            # -------------------------------------------------
                            # Cell activation logic/actions
                            image_stable, max_count = self.max_overlap(tol=1.0)
                            
                            # Debug dot at origin
                            dot_color = (0, 255, 0)  if image_stable else (255, 0, 0)
                            cv2.circle(overlay, (int(origin_imgpt[0]), int(origin_imgpt[1])), 12, dot_color, -1)
                            print(f"History: {len(self.board_corners_history)}, Overlap: {max_count}")

                            if not self.focus_active:
                                if not image_stable:
                                    _, cell_id = self.is_at_cell(origin_imgpt)
                                    self.current_r, self.current_c = cell_id

                                else:
                                    success, cell_id = self.activate_cell_at_point(origin_imgpt)    # Activate circle if origin is inside one
                                    if success:
                                        print(f"{Colors().text(str(cell_id), 'green')}")
                                        
                                        # Capture and save an image sized like the on-screen feed
                                        r, c = cell_id
                                        filename = f"cell_{r}_{c}.png"
                                        filepath = os.path.join(self.calib_path, filename)

                                        save_rgb = img_array
                                        target_w = self._feed_w if self._feed_w else w
                                        target_h = self._feed_h if self._feed_h else h

                                        if target_w != w or target_h != h:
                                            interp = cv2.INTER_AREA if target_w < w or target_h < h else cv2.INTER_LINEAR
                                            save_rgb = cv2.resize(save_rgb, (target_w, target_h), interpolation=interp)

                                        cv2.imwrite(filepath, cv2.cvtColor(save_rgb, cv2.COLOR_RGB2BGR))
                                        # print(f"Saved image: {filepath}")
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

                if not pose_detected and getattr(self.gl, "pose_valid", False):
                    self.gl.clear_board_pose()

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
