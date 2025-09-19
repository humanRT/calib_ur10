# utils.py

import os
import shutil
from PyQt5.QtWidgets import QMessageBox, QApplication

class Colors:
    """
    Utility for printing colored text in the console using ANSI escape codes.

    Usage:
        from utils import Colors

        # Print colored text
        print(Colors.text("Error!", "red"))
        print(Colors.text("Success!", "green"))
        print(Colors.text("Warning!", "yellow"))

        # Use raw color codes if needed
        print(Colors.RED + "Direct ANSI code usage" + Colors.RESET)

    Supported colors:
        red, green, yellow, blue, magenta, cyan, white, reset
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    @staticmethod
    def text(text: str, color: str) -> str:
        color_map = {
            "red": Colors.RED,
            "green": Colors.GREEN,
            "yellow": Colors.YELLOW,
            "blue": Colors.BLUE,
            "magenta": Colors.MAGENTA,
            "cyan": Colors.CYAN,
            "white": Colors.WHITE,
            "reset": Colors.RESET,
        }
        return f"{color_map.get(color, Colors.RESET)}{text}{Colors.RESET}"


class FileHelper:
    def __init__(self, parent=None):
        """
        parent: QWidget parent for QMessageBox dialogs
        """
        self.parent = parent

    def create_directory(self, path: str, force_replace: bool = True) -> bool:
        """
        Create a directory.

        Behavior:
        - If force_replace=True and the directory exists, it will be removed and recreated automatically.
        - If force_replace=False and the directory exists, a popup will ask whether to replace or abort.

        Returns:
            True if directory was created or replaced
            False if operation aborted or failed
        """
        if os.path.exists(path):
            if force_replace:
                # Auto replace without asking
                try:
                    shutil.rmtree(path)
                    os.makedirs(path)
                    return True
                except Exception as e:
                    self._show_warning(f"Failed to replace directory:\n{e}")
                    return False
            else:
                # Ask the user
                choice = QMessageBox.question(
                    self.parent,
                    "Directory Exists",
                    f"Directory '{path}' already exists.\n\nDo you want to replace it?",
                    QMessageBox.Yes | QMessageBox.Abort,
                    QMessageBox.Abort,
                )
                if choice == QMessageBox.Yes:
                    try:
                        shutil.rmtree(path)
                        os.makedirs(path)
                        return True
                    except Exception as e:
                        self._show_warning(f"Failed to replace directory:\n{e}")
                        return False
                else:
                    return False
        else:
            try:
                os.makedirs(path)
                return True
            except Exception as e:
                self._show_warning(f"Failed to create directory:\n{e}")
                return False

    def _show_warning(self, message: str):
        """Internal helper to show a warning message box."""
        QMessageBox.warning(self.parent, "Warning", message)