import cv2
import numpy as np


class FocusAssistant:
    def __init__(self, hysteresis=100):
        self.prev_score = None
        self.best_score = 1e-6      # avoid div by zero
        self.best_norm = 0.0        # track best normalized score
        self.history = []          # store last N improvements
        self.hysteresis = hysteresis

    def focus_score(self, img):
        """Return focus sharpness score based on Laplacian variance."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def get_focus_info(self, img):
        """
        Compute normalized focus quality and directional hint.

        Returns:
            norm: float (0.0 to 1.0) normalized focus quality
            improving: bool (True if score improved since last frame)
        """
        score = self.focus_score(img)
        self.best_score = max(self.best_score, score)

        norm = min(score / self.best_score, 1.0)
        self.best_norm = max(self.best_norm, norm)

        improving = True
        if self.prev_score is not None:
            improving = score > self.prev_score

        # Keep sliding history of True/False
        self.history.append(improving)
        if len(self.history) > self.hysteresis:
            self.history.pop(0)

        # Only change status if majority of last N frames agree
        improving_stable = sum(self.history) > len(self.history) // 2

        if self.prev_score is None:
            hint = "Start adjusting focus"
        elif improving_stable:
            hint = f"Improving ({score:.0f})"
        else:
            hint = f"Worsening ({score:.0f})"

        self.prev_score = score
        return norm, hint, improving_stable

    def draw_focus_bar(self, img, norm, hint, improving=True):
        """
        Draw a focus bar with directional hint and peak marker.
        """
        h, w, _ = img.shape
        bar_w, bar_h = int(w * 0.6), 30
        x0, y0 = int((w - bar_w) / 2), h - 60
        x1, y1 = x0 + int(bar_w * norm), y0 + bar_h

        # Background bar (gray)
        cv2.rectangle(img, (x0, y0), (x0 + bar_w, y1), (50, 50, 50), -1)

        # Foreground bar
        color = (0, 255, 0) if improving else (0, 0, 255)
        cv2.rectangle(img, (x0, y0), (x1, y1), color, -1)

        # Peak marker (white tick at best focus so far)
        peak_x = x0 + int(bar_w * self.best_norm)
        cv2.line(img, (peak_x, y0), (peak_x, y1), (255, 255, 255), 2)

        # Hint text
        cv2.putText(
            img,
            hint,
            (x0, y0 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
