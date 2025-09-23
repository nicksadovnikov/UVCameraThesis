import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from pathlib import Path
import subprocess
import shutil
from datetime import datetime

app = Flask(__name__)

# Default folders
DEFAULT_FRAME_DIR = Path("static/captures/all_captures")
DEFAULT_RESULTS_DIR = Path("static/captures/results")
DEFAULT_FRAME_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def capture_images(wavelength_nm, shutter_ms, frame_count, out_dir):
    """Capture raw DNG frames with libcamera-still into the chosen folder"""
    shutter_us = shutter_ms * 1000
    for i in range(frame_count):
        base = f"{wavelength_nm}nm_{shutter_ms}ms_frame{i+1:03d}"
        jpg_path = out_dir / f"{base}.jpg"   # must be .jpg for libcamera
        cmd = [
            "libcamera-still",
            "--gain", "1",
            "--shutter", str(shutter_us),
            "--raw",              # ensures a .dng is also created
            "-o", str(jpg_path),
            "-t", "1000"
        ]
        subprocess.run(cmd, check=True)


def stack_images(wavelength_nm, shutter_ms, raw_dir, result_dir, method="average"):
    """Stack only the DNG frames and save .dng + .jpg preview"""
    images = []
    dng_files = sorted(raw_dir.glob(f"{wavelength_nm}nm_{shutter_ms}ms_frame*.dng"))

    for f in dng_files:
        raw = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if raw is None:
            continue
        images.append(raw.astype(np.float32))

    if not images:
        raise ValueError("No .dng frames found to stack.")

    if method == "average":
        stacked = np.mean(images, axis=0)
    elif method == "median":
        stacked = np.median(np.array(images), axis=0)
    else:
        raise ValueError("Invalid stacking method")

    stacked = np.clip(stacked, 0, 65535).astype(np.uint16)

    # Timestamped result folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    result_subdir = result_dir / timestamp
    result_subdir.mkdir(parents=True, exist_ok=True)

    # Save stacked result as DNG
    dng_path = result_subdir / f"{wavelength_nm}nm_{shutter_ms}ms_stacked.dng"
    cv2.imwrite(str(dng_path), stacked)

    # Create 8-bit preview
    preview = (stacked / 256).astype(np.uint8)

    # Highlight saturated pixels (≥ 250 in 8-bit scale)
    mask = preview >= 250
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    preview_bgr[mask] = [0, 0, 255]  # red overlay

    jpg_path = result_subdir / f"{wavelength_nm}nm_{shutter_ms}ms_stacked.jpg"
    cv2.imwrite(str(jpg_path), preview_bgr)

    return jpg_path.relative_to("static")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    wavelength_nm = int(request.form["wavelength"])
    shutter_ms = int(request.form["shutter"])
    frame_count = int(request.form["frames"])
    method = request.form.get("stack_method", "average")

    # Use user-specified directories or defaults
    frame_folder = Path(request.form["frame_folder"] or DEFAULT_FRAME_DIR)
    result_folder = Path(request.form["result_folder"] or DEFAULT_RESULTS_DIR)

    frame_folder.mkdir(parents=True, exist_ok=True)
    result_folder.mkdir(parents=True, exist_ok=True)

    # Capture frames into selected/all_captures folder
    capture_images(wavelength_nm, shutter_ms, frame_count, frame_folder)

    # Stack results into results/<timestamp>/
    result_rel = stack_images(wavelength_nm, shutter_ms, frame_folder, result_folder, method)

    return render_template(
        "result.html",
        result_image=str(result_rel),
        session_dir=str(result_folder)
    )


@app.route("/download")
def download():
    """Bundle everything under static/captures into a zip"""
    zip_path = "experiment_results.zip"
    shutil.make_archive("experiment_results", 'zip', "static/captures")
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
