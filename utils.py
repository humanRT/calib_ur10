# utils.py
class colors:
    """
    Utility for printing colored text in the console using ANSI escape codes.

    Usage:
        from utils import colors

        # Print colored text
        print(colors.text("Error!", "red"))
        print(colors.text("Success!", "green"))
        print(colors.text("Warning!", "yellow"))

        # Use raw color codes if needed
        print(colors.RED + "Direct ANSI code usage" + colors.RESET)

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
            "red": colors.RED,
            "green": colors.GREEN,
            "yellow": colors.YELLOW,
            "blue": colors.BLUE,
            "magenta": colors.MAGENTA,
            "cyan": colors.CYAN,
            "white": colors.WHITE,
            "reset": colors.RESET,
        }
        return f"{color_map.get(color, colors.RESET)}{text}{colors.RESET}"
