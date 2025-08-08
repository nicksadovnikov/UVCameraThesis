# File: app.py

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from datetime import datetime
import subprocess
from pathlib import Path

app = Flask(__name__)

# Directory to store captured and processed images
CAPTURE_DIR = Path("static/captures")
PROCESSED_IMAGE = CAPTURE_DIR / "stacked_result.png"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)


def capture_images(shutter_us, frame_count):
    """
    Captures a number of frames using libcamera-still with RAW output.
    Each frame is saved as both .jpg and .raw in the capture directory.
    """
    for i in range(frame_count):
        filename = f"frame_{i:03d}.jpg"
        filepath = CAPTURE_DIR / filename
        cmd = [
            "libcamera-still",
            f"--shutter", str(shutter_us),
            "--gain", "1",
            "--raw",
            "-o", str(filepath),
            "-t", "1000"
        ]
        subprocess.run(cmd, check=True)


def extract_blue_channel(image_path):
    """Extracts the blue channel from a JPG image."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img[:, :, 0]  # Blue channel


def stack_images(method="average"):
    """
    Stacks the blue channels of all .jpg images in the capture directory.
    method: 'average' or 'median'
    """
    images = []
    for img_file in sorted(CAPTURE_DIR.glob("frame_*.jpg")):
        blue = extract_blue_channel(img_file)
        images.append(blue.astype(np.float32))

    if not images:
        raise ValueError("No images found to stack.")

    if method == "average":
        stacked = np.mean(images, axis=0)
    elif method == "median":
        stacked = np.median(np.array(images), axis=0)
    else:
        raise ValueError("Invalid stacking method.")

    stacked = np.clip(stacked, 0, 255).astype(np.uint8)

    # Apply histogram equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stacked)

    cv2.imwrite(str(PROCESSED_IMAGE), enhanced)
    return PROCESSED_IMAGE


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    shutter_ms = int(request.form["shutter"])
    frame_count = int(request.form["frames"])
    method = request.form.get("stack_method", "average")

    # Convert to microseconds
    shutter_us = shutter_ms * 1000

    # Clear previous captures
    for f in CAPTURE_DIR.glob("frame_*.jpg"):
        f.unlink()
    for f in CAPTURE_DIR.glob("frame_*.jpg.raw"):
        f.unlink()

    capture_images(shutter_us, frame_count)
    result_path = stack_images(method)
    return render_template("result.html", result_image=result_path.name)


@app.route("/download")
def download():
    return send_file(PROCESSED_IMAGE, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
