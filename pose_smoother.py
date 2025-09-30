import time
import numpy as np
import cv2

class PoseSmoother:
    def __init__(self, alpha: float = 0.8, timeout: float = 0.3):
        """
        alpha   = smoothing factor (0 < alpha < 1). Higher → smoother but laggier.
        timeout = seconds to keep using last good pose after detection loss.
        """
        self.alpha = alpha
        self.timeout = timeout
        self.last_update = 0.0
        self.smoothed_rvec = None
        self.smoothed_tvec = None

    def update(self, rvec: np.ndarray, tvec: np.ndarray):
        """Feed a new valid pose into the smoother."""
        rvec = np.asarray(rvec, dtype=np.float32).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float32).reshape(3, 1)

        if self.smoothed_rvec is None:
            self.smoothed_rvec = rvec.copy()
            self.smoothed_tvec = tvec.copy()
        else:
            self.smoothed_rvec = (self.alpha * rvec + (1 - self.alpha) * self.smoothed_rvec)
            self.smoothed_tvec = (self.alpha * tvec + (1 - self.alpha) * self.smoothed_tvec) # type: ignore

        self.last_update = time.time()

    def get_pose(self):
        """Return the current smoothed pose or None if timeout exceeded."""
        if (
            self.smoothed_rvec is not None
            and time.time() - self.last_update < self.timeout
        ):
            return self.smoothed_rvec, self.smoothed_tvec
        return None, None
