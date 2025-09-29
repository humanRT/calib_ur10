#!/usr/bin/env python3
"""
Charuco camera calibration with OpenCV (single camera)

Features
- Generate and save a printable Charuco board PNG at true scale
- Collect calibration views from a camera or a folder of images
- Detect ArUco markers, interpolate Charuco corners, and calibrate
- Report RMS reprojection error and per–view errors
- Save intrinsics to YAML and show undistortion preview

Requirements
    pip install opencv-contrib-python numpy

Usage examples
1) Make a board for printing at 200 DPI, A3-ish size:
   python charuco_calibration.py --draw-board board.png \
       --squares-x 12 --squares-y 9 --square 0.020 --marker 0.016 --dpi 200

   Meaning: square size 20 mm, marker size 16 mm. Print at 100% scale on matte paper.

2) Calibrate from a live camera (press C to capture a good view, Q to finish):
   python charuco_calibration.py --from-camera 0 --preview  \
       --squares-x 12 --squares-y 9 --square 0.020 --marker 0.016 \
       --out calib_charuco.yaml

3) Calibrate from a folder of JPEGs/PNGs:
   python charuco_calibration.py --from-folder ./calib_imgs --preview \
       --squares-x 12 --squares-y 9 --square 0.020 --marker 0.016 \
       --out calib_charuco.yaml

Notes
- Use at least 20 to 30 diverse views, with the board at different positions, scales, and tilts.
- Units are meters. If you prefer millimeters, divide by 1000 in the flags.
- Ensure the dictionary you draw for the board matches the one used for detection.
"""

import argparse
import glob
import os
import sys

import cv2
import numpy as np

from pathlib import Path
from PIL import Image
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas

# OpenCV ArUco API lives in contrib build
try:
    aruco = cv2.aruco
except AttributeError as e:
    raise SystemExit("This script needs opencv-contrib-python. Install with: pip install opencv-contrib-python")


def get_dict(dict_name: str):
    """Return a cv2.aruco dictionary from a friendly name."""
    name = dict_name.upper()
    if not name.startswith("DICT_"):
        name = "DICT_" + name
    if not hasattr(aruco, name):
        raise ValueError(f"Unknown dictionary '{dict_name}'. Examples: DICT_5X5_100, DICT_4X4_250, DICT_6X6_50")
    return aruco.getPredefinedDictionary(getattr(aruco, name))


def make_board(squares_x: int, squares_y: int, square: float, marker: float, dict_name: str):
    """Create a CharucoBoard object.
    squares_x, squares_y are number of internal chessboard squares (not markers).
    square and marker are side lengths in meters.
    """

    print (squares_x, squares_y, square, marker, dict_name)

    dictionary = get_dict(dict_name)
    board = aruco.CharucoBoard((squares_x, squares_y), squareLength=square, markerLength=marker, dictionary=dictionary)
    return board, dictionary


def draw_board(board, out_path: str, dpi: int = 200):
    """Render the Charuco board to a PNG that prints at the given DPI.
    The image size is chosen to fit the full board with a small white border.
    """
    px_per_meter = dpi / 25.4 * 1000.0  # pixels per meter at given DPI
    width_m = (board.getChessboardSize()[0]) * board.getSquareLength()
    height_m = (board.getChessboardSize()[1]) * board.getSquareLength()

    w_px = int(round(width_m * px_per_meter))
    h_px = int(round(height_m * px_per_meter))

    # Add a quiet white border around the board for easier detection
    border = int(0.03 * max(w_px, h_px))
    img = board.generateImage((w_px, h_px), marginSize=border)

    ok = cv2.imwrite(out_path, img)
    if not ok:
        raise IOError(f"Failed to write {out_path}")


def draw_board_pdf(board, out_path: str, dpi: int = 200, page_size: str = "letter"):
    """Render the Charuco board to a PDF at true physical size, with a 100 mm calibration line."""

    # Pick page size
    page_size = page_size.lower()
    if page_size == "a4":
        page_w, page_h = A4
    else:
        page_w, page_h = letter

    # Convert board size to pixels at given dpi
    px_per_meter = dpi / 25.4 * 1000.0
    width_m = board.getChessboardSize()[0] * board.getSquareLength()
    height_m = board.getChessboardSize()[1] * board.getSquareLength()
    w_px = int(round(width_m * px_per_meter))
    h_px = int(round(height_m * px_per_meter))

    border = int(0.03 * max(w_px, h_px))
    img = board.generateImage((w_px, h_px), marginSize=border)

    # Save temporary PNG
    tmp_png = out_path + ".tmp.png"
    cv2.imwrite(tmp_png, img)

    # Convert to inches
    w_in = w_px / dpi
    h_in = h_px / dpi

    # Create PDF
    c = canvas.Canvas(out_path, pagesize=(page_w, page_h))
    x0 = (page_w - w_in * inch) / 2
    y0 = (page_h - h_in * inch) / 2

    # Draw board at exact physical size
    c.drawImage(tmp_png, x0, y0, width=w_in * inch, height=h_in * inch,
                preserveAspectRatio=False, mask='auto')

    # Add 100 mm calibration line below the board
    line_len_mm = 100.0
    line_len_in = line_len_mm / 25.4
    line_x0 = x0
    line_x1 = x0 + line_len_in * inch
    line_y = y0 - 0.5 * inch  # half an inch below the board

    c.setLineWidth(1)
    c.line(line_x0, line_y, line_x1, line_y)
    c.drawString(line_x0, line_y - 12, "0 mm")
    c.drawRightString(line_x1, line_y - 12, "100 mm")

    c.save()
    os.remove(tmp_png)
    print(f"Saved board PDF to {out_path}. Print at 100% scale. Check the 100 mm line for accuracy.")


def detect_charuco(img_bgr, board, dictionary, refine: bool = True):
    """Detect ArUco markers and interpolate Charuco corners.
    Returns (charuco_corners, charuco_ids, vis_image) or (None, None, vis_image) if insufficient corners.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    params = aruco.DetectorParameters()
    if refine:
        refine_params = aruco.RefineParameters()
    else:
        refine_params = None

    detector = aruco.ArucoDetector(dictionary, params)
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None and len(ids) > 0 and refine_params is not None:
        aruco.refineDetectedMarkers(
            image=gray,
            board=board,
            detectedCorners=corners,
            detectedIds=ids,
            rejectedCorners=rejected,
            cameraMatrix=None,
            distCoeffs=None,
            parameters=refine_params,
        ) # type: ignore

    vis = img_bgr.copy()
    if ids is not None and len(ids) > 0:
        aruco.drawDetectedMarkers(vis, corners, ids)
        # Interpolate Charuco corners using subpixel accuracy for better calibration
        retval, ch_corners, ch_ids = aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=board,
            cameraMatrix=None,
            distCoeffs=None,
        )
        if retval is not None and ch_ids is not None and len(ch_ids) >= 8:
            # Draw the interpolated chessboard corners
            aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids)
            return ch_corners, ch_ids, vis

    return None, None, vis


def calibrate_charuco(all_corners, all_ids, board, image_size):
    """Run Charuco calibration and return results dict."""
    flags = 0
    # You can enable flags like cv2.CALIB_RATIONAL_MODEL for strong distortion lenses
    # flags |= cv2.CALIB_RATIONAL_MODEL

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    retval, K, D, rvecs, tvecs, std_int, std_ext, per_view = aruco.calibrateCameraCharucoExtended(
        charucoCorners=all_corners,
        charucoIds=all_ids,
        board=board,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
        flags=flags,
        criteria=criteria,
    ) # type: ignore

    return {
        "rms": float(retval),
        "K": K,
        "D": D,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "std_intr": std_int,
        "std_extr": std_ext,
        "per_view_err": per_view,
    }


def save_yaml(path, calib, image_size, board_cfg):
    fs = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise IOError(f"Cannot open {path} for writing")

    fs.write("image_width", int(image_size[0]))
    fs.write("image_height", int(image_size[1]))
    fs.write("camera_matrix", calib["K"])  # 3x3
    fs.write("distortion_coefficients", calib["D"])  # 1x5 or 1x8 depending on flags
    fs.write("rms", float(calib["rms"]))

    fs.startWriteStruct("board", cv2.FileNode_MAP)
    fs.write("squares_x", int(board_cfg["sx"]))
    fs.write("squares_y", int(board_cfg["sy"]))
    fs.write("square_length_m", float(board_cfg["square"]))
    fs.write("marker_length_m", float(board_cfg["marker"]))
    fs.write("dictionary", board_cfg["dict"]) 
    fs.endWriteStruct()

    fs.release()


def undistort_demo(img_bgr, K, D):
    h, w = img_bgr.shape[:2]
    newK, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=0.0)  # alpha 0 for full crop
    und = cv2.undistort(img_bgr, K, D, None, newK)
    x, y, rw, rh = roi
    if rw > 0 and rh > 0:
        und = und[y:y+rh, x:x+rw]
    return und


def collect_from_camera(cam_spec: str, board, dictionary, preview: bool):
    # cam_spec can be an integer index or a URL. Try int first
    cap = None
    try:
        idx = int(cam_spec)
        cap = cv2.VideoCapture(idx)
    except ValueError:
        cap = cv2.VideoCapture(cam_spec)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera '{cam_spec}'")

    print("Press C to capture a view, Q to finish, or SPACE to skip")
    samples = []
    last_vis = None
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break

        ch_corners, ch_ids, vis = detect_charuco(frame, board, dictionary)
        last_vis = vis
        if preview:
            cv2.imshow("Charuco detect", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), ord('Q')):
            break
        if key in (ord('c'), ord('C')) and ch_ids is not None:
            samples.append((ch_corners, ch_ids, frame.shape[1], frame.shape[0]))
            print(f"Captured view {len(samples)} with {len(ch_ids)} Charuco corners")

    cap.release()
    if preview:
        cv2.destroyAllWindows()

    return samples, last_vis


def collect_from_folder(folder: str, board, dictionary, preview: bool):
    print(f"Hello {folder}")
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    files = []
    for e in exts:
        files.extend(glob.glob(str(Path(folder) / e)))
    files = sorted(files)
    if not files:
        raise RuntimeError(f"No images found in {folder}")

    samples = []
    last_vis = None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            print(f"Warning: could not read {f}")
            continue
        ch_corners, ch_ids, vis = detect_charuco(img, board, dictionary)
        last_vis = vis
        if preview:
            cv2.imshow("Charuco detect", vis)
            cv2.waitKey(200)
        if ch_ids is not None:
            samples.append((ch_corners, ch_ids, img.shape[1], img.shape[0]))
            print(f"{Path(f).name}: {len(ch_ids)} Charuco corners")
        else:
            print(f"{Path(f).name}: detection failed")

    if preview:
        cv2.destroyAllWindows()

    return samples, last_vis


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--squares-x", type=int, default=12, help="Charuco chessboard squares along X")
    p.add_argument("--squares-y", type=int, default=9, help="Charuco chessboard squares along Y")
    p.add_argument("--square", type=float, default=0.020, help="Square size in meters")
    p.add_argument("--marker", type=float, default=0.016, help="Marker size in meters")
    p.add_argument("--dict", type=str, default="DICT_5X5_100", help="ArUco dictionary name")

    p.add_argument("--draw-board", type=str, help="Output PNG file for a printable board")
    p.add_argument("--dpi", type=int, default=200, help="DPI for board rendering")

    src = p.add_mutually_exclusive_group()
    src.add_argument("--from-camera", type=str, help="Camera index or URL")
    src.add_argument("--from-folder", type=str, help="Folder of calibration images")

    p.add_argument("--out", type=str, default="calib_charuco.yaml", help="YAML file for intrinsics")
    p.add_argument("--preview", action="store_true", help="Show detection preview windows")

    args = p.parse_args()

    board, dictionary = make_board(args.squares_x, args.squares_y, args.square, args.marker, args.dict)

    if args.draw_board:
        draw_board(board, args.draw_board, args.dpi)
        print(f"Saved board to {args.draw_board}. Print at 100% scale.")

        # Also export PDF for reliable printing
        pdf_path = Path(args.draw_board).with_suffix(".pdf")
        draw_board_pdf(board, str(pdf_path), args.dpi)

        if not args.from_camera and not args.from_folder:
            return

    # Collect views
    if args.from_camera:
        samples, last_vis = collect_from_camera(args.from_camera, board, dictionary, args.preview)
    elif args.from_folder:
        samples, last_vis = collect_from_folder(args.from_folder, board, dictionary, args.preview)
    else:
        print("No images or camera specified. Use --from-camera or --from-folder, or just --draw-board to generate a board.")
        return

    if not samples:
        print("No valid detections. Nothing to calibrate.")
        return

    # Prepare inputs for calibration
    all_corners = []
    all_ids = []
    sizes = set()
    for ch_corners, ch_ids, w, h in samples:
        all_corners.append(ch_corners)
        all_ids.append(ch_ids)
        sizes.add((w, h))

    if len(sizes) != 1:
        print("Error: images have different sizes. Use consistent resolution for all views.")
        return

    image_size = next(iter(sizes))

    calib = calibrate_charuco(all_corners, all_ids, board, image_size)

    # Report errors
    per_view = calib["per_view_err"].ravel() if calib["per_view_err"] is not None else np.array([])
    if per_view.size:
        print(f"RMS reprojection error: {calib['rms']:.4f} px")
        print(f"Per view error: mean {per_view.mean():.4f} px, median {np.median(per_view):.4f} px, max {per_view.max():.4f} px over {len(per_view)} views")
    else:
        print(f"RMS reprojection error: {calib['rms']:.4f} px")

    # Save YAML
    save_yaml(args.out, calib, image_size, {
        "sx": args.squares_x,
        "sy": args.squares_y,
        "square": args.square,
        "marker": args.marker,
        "dict": args.dict,
    })
    print(f"Saved intrinsics to {args.out}")

    # Show undistortion using the last preview image if available
    if last_vis is not None and calib["K"] is not None:
        und = undistort_demo(last_vis, calib["K"], calib["D"])
        cv2.imshow("Undistorted preview (cropped)", und)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
