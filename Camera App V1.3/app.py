import os
import cv2
import numpy as np
from flask import Flask, render_template, request, send_file
from datetime import datetime
import subprocess
from pathlib import Path
import shutil

app = Flask(__name__)

BASE_DIR = Path("static/captures")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamped_dir():
    """Create a timestamped directory for this capture session."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = BASE_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def capture_images(shutter_us, frame_count, session_dir):
    """
    Captures a number of frames using libcamera-still with RAW output.
    Saves into a timestamped session directory.
    """
    for i in range(frame_count):
        filename = f"frame_{i:03d}.jpg"
        filepath = session_dir / filename
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


def stack_images(session_dir, method="average"):
    """
    Stacks the blue channels of all .jpg images in the session directory.
    Saves final image as stacked_result.png.
    """
    images = []
    for img_file in sorted(session_dir.glob("frame_*.jpg")):
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

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(stacked)

    result_path = session_dir / "stacked_result.png"
    cv2.imwrite(str(result_path), enhanced)
    return result_path


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    shutter_ms = int(request.form["shutter"])
    frame_count = int(request.form["frames"])
    method = request.form.get("stack_method", "average")

    shutter_us = shutter_ms * 1000
    session_dir = get_timestamped_dir()

    capture_images(shutter_us, frame_count, session_dir)
    result_path = stack_images(session_dir, method)

    return render_template("result.html", result_image=str(result_path.relative_to("static")), session_dir=str(session_dir.relative_to("static")))


@app.route("/download/<path:session_dir>")
def download(session_dir):
    zip_path = f"{session_dir}.zip"
    abs_dir = BASE_DIR / session_dir
    shutil.make_archive(str(abs_dir), 'zip', str(abs_dir))
    return send_file(f"static/captures/{zip_path}", as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)