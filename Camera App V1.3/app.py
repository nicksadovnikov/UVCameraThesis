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
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = BASE_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def capture_images(shutter_us, frame_count, session_dir):
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
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return img[:, :, 0]


def stack_images(session_dir, method="average"):
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

    rel_result = result_path.relative_to("static")
    rel_dir = session_dir.relative_to("static")

    return render_template("result.html", result_image=str(rel_result), session_dir=str(rel_dir))


@app.route("/download/<path:session_dir>")
def download(session_dir):
    abs_dir = BASE_DIR / session_dir
    zip_path = abs_dir.with_suffix(".zip")
    shutil.make_archive(str(abs_dir), 'zip', str(abs_dir))
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
