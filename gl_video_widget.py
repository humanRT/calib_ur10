from typing import Optional
import numpy as np
from PyQt5.QtCore import QSize
from PyQt5.QtWidgets import QOpenGLWidget
from OpenGL.GL import *  # type: ignore
from stl_utils import load_stl_with_open3d


class GLVideoWidget(QOpenGLWidget):
    """
    OpenGL widget for displaying video frames, board overlays, and STL meshes.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame: Optional[np.ndarray] = None
        self.texture_id: Optional[int] = None
        self._w, self._h = 1, 1
        self._frame_w, self._frame_h = 1280, 1024
        self.board_corners_history: list[np.ndarray] = []
        self.max_history = 20

        # STL mesh data
        self.mesh_vertices = None
        self.mesh_triangles = None
        self.mesh_normals = None

    def load_stl(self, filepath: str) -> None:
        verts, tris, norms = load_stl_with_open3d(filepath)
        self.mesh_vertices, self.mesh_triangles, self.mesh_normals = verts, tris, norms
        self.update()

    def set_board_corners(self, pts: np.ndarray) -> None:
        self.board_corners_history.append(np.asarray(pts, dtype=np.float32))
        if len(self.board_corners_history) > self.max_history:
            self.board_corners_history.pop(0)
        self.update()

    def sizeHint(self) -> QSize:
        return QSize(self._frame_w, self._frame_h)

    def set_frame(self, rgb_frame: np.ndarray) -> None:
        self.frame = rgb_frame
        h, w, _ = rgb_frame.shape
        if (w, h) != (self._frame_w, self._frame_h):
            self._frame_w, self._frame_h = w, h
            self.updateGeometry()
        self.update()

    # ---------------- OpenGL lifecycle ----------------
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

    def paintGL(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)
        if self.frame is None:
            return

        h, w, _ = self.frame.shape

        # Upload frame as texture
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, self.frame)

        # Draw full-screen textured quad
        glColor3f(1.0, 1.0, 1.0)
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0)
        glTexCoord2f(1.0, 1.0); glVertex2f( 1.0, -1.0)
        glTexCoord2f(1.0, 0.0); glVertex2f( 1.0,  1.0)
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0,  1.0)
        glEnd()

        # Draw STL mesh if loaded
        if self.mesh_vertices is not None:
            self._draw_stl()

        # Draw calibration board corners
        if self.board_corners_history:
            glDisable(GL_TEXTURE_2D)
            glLineWidth(2.0)

            n = len(self.board_corners_history)
            for i, corners in enumerate(self.board_corners_history):
                alpha = (i + 1) / n
                glColor4f(1.0, 0.0, 0.0, alpha)
                glBegin(GL_LINE_LOOP)
                for (px, py) in corners:
                    ndc_x = (px / w) * 2.0 - 1.0
                    ndc_y = 1.0 - (py / h) * 2.0
                    glVertex2f(ndc_x, ndc_y)
                glEnd()

            glEnable(GL_TEXTURE_2D)

    def _draw_stl(self) -> None:
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_TEXTURE_2D)
        glColor3f(0.7, 0.7, 0.9)

        glPushMatrix()
        glScalef(0.01, 0.01, 0.01)
        glTranslatef(0, 0, -5)

        glBegin(GL_TRIANGLES)
        for tri, n in zip(self.mesh_triangles, self.mesh_normals):  # type: ignore
            glNormal3fv(n)
            for idx in tri:
                glVertex3fv(self.mesh_vertices[idx])  # type: ignore
        glEnd()

        glPopMatrix()
        glEnable(GL_TEXTURE_2D)
        glDisable(GL_DEPTH_TEST)
