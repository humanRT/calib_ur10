import cv2
import numpy as np
from collections import  Counter
from typing import Optional
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QOpenGLWidget, QSizePolicy
from OpenGL.GL import *   # type: ignore
from OpenGL.GLU import *
from stl_utils import load_stl_file


class GLVideoWidget(QOpenGLWidget):
    """
    OpenGL widget for displaying video frames, board overlays, and STL meshes,
    and for rendering 3D geometry registered to the ChArUco board pose.
    """
    # ---------- Qt / GL lifecycle ----------
    def __init__(self, parent=None):
        super().__init__(parent)
        # video
        self.frame: Optional[np.ndarray] = None
        self.texture_id: Optional[int] = None
        self._w, self._h = 1, 1
        self._frame_w, self._frame_h = 1280, 1024
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedSize(self._frame_w, self._frame_h)
        # overlay mode and histories
        self.overlay_mode: str = "ideal"  # "ideal" or "image"
        self.ideal_corners_history: list[np.ndarray] = []
        self.image_corners_history: list[np.ndarray] = []
        # self.history_size: int = 20
        self.max_history = 20
        # STL mesh
        self.mesh_vertices = None
        self.mesh_triangles = None
        self.mesh_normals = None
        # camera + board
        self.camera_matrix: Optional[np.ndarray] = None
        self.dist_coeffs: Optional[np.ndarray] = None
        self.squaresX: Optional[int] = None
        self.squaresY: Optional[int] = None
        self.square_len: Optional[float] = None
        self.rvec: Optional[np.ndarray] = None
        self.tvec: Optional[np.ndarray] = None
        self.pose_valid: bool = False

    def sizeHint(self) -> QSize:
        return QSize(self._frame_w, self._frame_h)

    def initializeGL(self) -> None:
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

    def resizeGL(self, w: int, h: int) -> None:
        self._w = max(1, w)
        self._h = max(1, h)
        glViewport(0, 0, self._w, self._h)

    # ---------- public API you will call from VideoApp ----------
    def set_camera_calibration(self, camera_matrix: np.ndarray, dist_coeffs: np.ndarray | None = None) -> None:
        self.camera_matrix = camera_matrix.astype(np.float32).copy()
        self.dist_coeffs = None if dist_coeffs is None else dist_coeffs.astype(np.float32).copy()
        self.update()

    def configure_board(self, squaresX: int, squaresY: int, square_len: float) -> None:
        self.squaresX = int(squaresX)
        self.squaresY = int(squaresY)
        self.square_len = float(square_len)
        self.update()

    def set_board_pose(self, rvec, tvec):
        self.pose_valid = True
        self.rvec = rvec
        self.tvec = tvec

    def set_overlay_mode(self, mode: str) -> None:
        assert mode in ("image", "ideal")
        self.overlay_mode = mode
        self.update()

    def clear_board_pose(self) -> None:
        self.pose_valid = False
        self.rvec = None
        self.tvec = None
        self.update()

    def push_image_corners(self, pts: np.ndarray) -> None:
        self.image_corners_history.append(np.asarray(pts, dtype=np.float32))
        if len(self.image_corners_history) > self.max_history:
            self.image_corners_history.pop(0)

    def push_ideal_corners(self, pts: np.ndarray) -> None:
        self.ideal_corners_history.append(np.asarray(pts, dtype=np.float32))
        if len(self.ideal_corners_history) > self.max_history:
            self.ideal_corners_history.pop(0)

    def get_active_corners(self) -> list[np.ndarray]:
        if self.overlay_mode == "image":
            return self.image_corners_history
        return self.ideal_corners_history

    def load_stl(self, filepath: str) -> None:
        verts, tris, norms = load_stl_file(filepath)
        self.mesh_vertices, self.mesh_triangles, self.mesh_normals = verts, tris, norms
        self.update()
    
    def set_frame(self, rgb_frame: np.ndarray) -> None:
        self.frame = rgb_frame
        h, w, _ = rgb_frame.shape
        if (w, h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = w, h
            self.setFixedSize(w, h)
            self.updateGeometry()
        self.update()
    
    # ---------- rendering ----------
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT) # type: ignore

        # -- Pass 1: 2D video + red corners overlay
        if self.frame is not None:
            self.draw_video_and_corners()            

        # Ensure a clean Z buffer before 3D
        glClear(GL_DEPTH_BUFFER_BIT)

        # -- Pass 2: 3D content registered to the board pose if we have one
        if self.pose_valid and self.camera_matrix is not None and self.squaresX and self.squaresY and self.square_len:
            self.draw_cube_on_plate()
            pass
        elif self.mesh_vertices is not None:
            # Fallback: free 3D view of STL if no pose
            self.draw_stl_freecam()

    # ---------- helpers: 2D pass ----------
    def draw_video_and_corners(self):
        h, w, _ = self.frame.shape # type: ignore

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(-1, 1, -1, 1, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # upload texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, self.frame)

        # video quad
        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(-1, -1)
        glTexCoord2f(1, 1); glVertex2f( 1, -1)
        glTexCoord2f(1, 0); glVertex2f( 1,  1)
        glTexCoord2f(0, 0); glVertex2f(-1,  1)
        glEnd()

         # pick which history to draw
        active_hist = self.ideal_corners_history if self.overlay_mode == "ideal" else self.image_corners_history
        if active_hist:
            glDisable(GL_TEXTURE_2D)
            glLineWidth(2.0)
            n = len(active_hist)
            for i, corners in enumerate(active_hist):
                alpha = (i + 1) / n
                glColor4f(1.0, 0.0, 0.0, alpha)
                glBegin(GL_LINE_LOOP)
                for (px, py) in corners:
                    ndc_x = (px / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (py / h) * 2.0
                    glVertex2f(ndc_x, ndc_y)
                glEnd()
            glEnable(GL_TEXTURE_2D)

        # restore
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    # ---------- helpers: 3D registered to camera ----------
    def load_projection_from_intrinsics(self, near: float = 0.01, far: float = 100.0):
        fx = float(self.camera_matrix[0, 0]) # type: ignore
        fy = float(self.camera_matrix[1, 1]) # type: ignore
        cx = float(self.camera_matrix[0, 2]) # type: ignore
        cy = float(self.camera_matrix[1, 2]) # type: ignore

        w = float(max(1, self._w))
        h = float(max(1, self._h))

        proj = np.zeros((4, 4), dtype=np.float32)
        proj[0, 0] = 2.0 * fx / w
        proj[1, 1] = 2.0 * fy / h
        proj[0, 2] = 1.0 - 2.0 * cx / w
        proj[1, 2] = 2.0 * cy / h - 1.0
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -2.0 * far * near / (far - near)
        proj[3, 2] = -1.0

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glLoadMatrixf(proj.T)

    def load_modelview_from_cv_pose(self):
        R, _ = cv2.Rodrigues(self.rvec)  # type: ignore # board→camera
        Rt = np.eye(4, dtype=np.float32)
        Rt[:3, :3] = R.astype(np.float32)
        Rt[:3, 3] = self.tvec.reshape(3).astype(np.float32) # type: ignore

        flip_yz = np.diag([1.0, -1.0, -1.0, 1.0])

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        glLoadMatrixf((flip_yz @ Rt).T)

    def draw_cube_on_plate(self):
        self.load_projection_from_intrinsics()
        self.load_modelview_from_cv_pose()

        board_w = float(self.squaresX) * float(self.square_len) # type: ignore
        board_h = float(self.squaresY) * float(self.square_len) # type: ignore
        cube_height = 0.5 * min(board_w, board_h)  # make it obvious

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

        glPushMatrix()
        # --- shift cube so it is centered on the board ---
        glTranslatef(board_w / 2.0, board_h / 2.0, 0.0)

        # Scale a unit cube [0,1]^3 to board footprint in X,Y and desired height in Z
        glScalef(board_w * 0.25, board_h * 0.25, cube_height)
        self.draw_unit_cube()
        glPopMatrix()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        # restore matrices
        glPopMatrix()            # MODELVIEW
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    def draw_unit_cube(self):
        glBegin(GL_QUADS)
        # Top z=0 (magenta) → sits on the plate
        glColor3f(1, 0, 1)
        glVertex3f(0, 0, 0); glVertex3f(0, 1, 0); glVertex3f(1, 1, 0); glVertex3f(1, 0, 0)
        # Bottom z=-1 (cyan) → extruded downward
        glColor3f(0, 1, 1)
        glVertex3f(0, 0, -1); glVertex3f(1, 0, -1); glVertex3f(1, 1, -1); glVertex3f(0, 1, -1)
        # Front y=1 (red)
        glColor3f(1, 0, 0)
        glVertex3f(0, 1, 0); glVertex3f(1, 1, 0); glVertex3f(1, 1, -1); glVertex3f(0, 1, -1)
        # Back y=0 (green)
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0); glVertex3f(0, 0, -1); glVertex3f(1, 0, -1); glVertex3f(1, 0, 0)
        # Left x=0 (blue)
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0); glVertex3f(0, 1, 0); glVertex3f(0, 1, -1); glVertex3f(0, 0, -1)
        # Right x=1 (yellow)
        glColor3f(1, 1, 0)
        glVertex3f(1, 0, 0); glVertex3f(1, 0, -1); glVertex3f(1, 1, -1); glVertex3f(1, 1, 0)
        glEnd()

    # ---------- optional: freecam STL draw if no pose ----------
    def draw_stl_freecam(self):
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        gluPerspective(45.0, max(1, self._w) / max(1, self._h), 0.1, 100.0)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        gluLookAt(3, 3, 5, 0, 0, 0, 0, 1, 0)

        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)

        glColor3f(0.7, 0.7, 0.9)
        glBegin(GL_TRIANGLES)
        for tri, n in zip(self.mesh_triangles, self.mesh_normals):  # type: ignore
            glNormal3fv(n)
            for idx in tri:
                glVertex3fv(self.mesh_vertices[idx])  # type: ignore
        glEnd()

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_TEXTURE_2D)

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
