import os
import cv2
import numpy as np
import subprocess
from flask import Flask, render_template, request, send_file
from datetime import datetime
from pathlib import Path
import shutil
import tifffile as tiff  # for saving .dng

app = Flask(__name__)

BASE_DIR = Path("static/captures")
BASE_DIR.mkdir(parents=True, exist_ok=True)


def get_timestamped_dir():
    """Create a new session folder with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = BASE_DIR / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def set_exposure_ms(exposure_ms: int):
    """
    Set manual exposure for OV9281.
    - auto_exposure must be set to 1 for manual mode
    - exposure_time_absolute is in units of 100 µs (1–5000)
    """
    # Convert ms → 100 µs units
    exposure_units = int(exposure_ms * 10)

    # Clip to valid range
    exposure_units = max(1, min(5000, exposure_units))

    subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0",
         "-c", "auto_exposure=1",
         "-c", f"exposure_time_absolute={exposure_units}"],
        check=True
    )

    return exposure_units



def capture_images(exposure_ms, frame_count, session_dir):
    """Capture N grayscale frames and save as .dng with exposure in filename."""
    set_exposure_ms(exposure_ms)

    cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    cam.set(cv2.CAP_PROP_FPS, 10)

    frames = []
    for i in range(frame_count):
        ret, frame = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(gray)   # keep as uint8

        frame_path = session_dir / f"{exposure_ms}ms_frame_{i:03d}.dng"
        tiff.imwrite(str(frame_path), gray.astype(np.uint8), photometric="minisblack")


    cam.release()
    return frames


def stack_images(frames, session_dir, exposure_ms):
    """Average stack frames and save as .dng + preview .jpg with exposure in filename."""
    stacked = np.mean(frames, axis=0).astype(np.uint8)

    dng_path = session_dir / f"{exposure_ms}ms_stacked_result.dng"
    tiff.imwrite(str(dng_path), stacked, photometric="minisblack")

    # Save JPEG preview (already 8-bit, just copy)
    preview_path = session_dir / f"{exposure_ms}ms_stacked_preview.jpg"
    cv2.imwrite(str(preview_path), stacked)


    return dng_path, preview_path



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    exposure_ms = int(request.form["shutter"])
    frame_count = int(request.form["frames"])

    session_dir = get_timestamped_dir()
    frames = capture_images(exposure_ms, frame_count, session_dir)
    dng_path, preview_path = stack_images(frames, session_dir, exposure_ms)

    rel_preview = preview_path.relative_to("static")
    rel_dir = session_dir.relative_to("static")

    return render_template(
        "result.html",
        result_image=str(rel_preview),
        session_dir=str(rel_dir)
    )



@app.route("/download/<path:session_dir>")
def download(session_dir):
    abs_dir = BASE_DIR / session_dir
    zip_path = abs_dir.with_suffix(".zip")
    shutil.make_archive(str(abs_dir), 'zip', str(abs_dir))
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
