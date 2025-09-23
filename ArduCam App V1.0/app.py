import os
import cv2
import numpy as np
import subprocess
from flask import Flask, render_template, request, send_file
from datetime import datetime
from pathlib import Path
import shutil
import tifffile as tiff  # for saving .dng
import json

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
    exposure_units = max(1, min(5000, exposure_units))

    subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0",
         "-c", "auto_exposure=1",
         "-c", f"exposure_time_absolute={exposure_units}"],
        check=True
    )

    return exposure_units


def get_camera_controls():
    """
    Query v4l2 controls and return as a dictionary {control: value}.
    """
    result = subprocess.run(
        ["v4l2-ctl", "-d", "/dev/video0", "--list-ctrls"],
        capture_output=True, text=True, check=True
    )
    controls = {}
    for line in result.stdout.splitlines():
        if "value=" in line:
            key = line.strip().split()[0]
            value_str = line.split("value=")[-1].split()[0]
            controls[key] = value_str
    return controls


def save_dng_with_metadata(filepath: Path, image: np.ndarray, metadata: dict):
    """
    Save grayscale image as .dng with metadata JSON embedded in ImageDescription.
    """
    description = json.dumps(metadata, indent=2)
    tiff.imwrite(
        str(filepath),
        image.astype(np.uint8),
        photometric="minisblack",
        description=description
    )


def capture_images(wavelength, exposure_ms, frame_count, save_dir):
    """Capture N grayscale frames and save with metadata embedded."""
    exposure_units = set_exposure_ms(exposure_ms)
    controls = get_camera_controls()

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
        frames.append(gray)

        filename = f"{wavelength}nm_{exposure_ms}ms_frame{i+1:03d}.dng"
        filepath = save_dir / filename

        metadata = {
            "Wavelength_nm": wavelength,
            "Exposure_ms": exposure_ms,
            "Exposure_units_100us": exposure_units,
            "FrameIndex": i + 1,
            "Timestamp": datetime.now().isoformat(),
            "CameraControls": controls
        }

        save_dng_with_metadata(filepath, gray, metadata)

    cam.release()
    return frames, controls


def stack_images(frames, stack_dir, wavelength, exposure_ms, controls):
    """Average frames, save stacked DNG, preview JPG, and highlight saturated pixels."""
    stacked = np.mean(frames, axis=0).astype(np.uint8)

    # Save DNG with metadata
    dng_name = f"{wavelength}nm_{exposure_ms}ms_stacked_result.dng"
    dng_path = stack_dir / dng_name
    metadata = {
        "Wavelength_nm": wavelength,
        "Exposure_ms": exposure_ms,
        "FrameCount": len(frames),
        "Timestamp": datetime.now().isoformat(),
        "CameraControls": controls
    }
    save_dng_with_metadata(dng_path, stacked, metadata)

    # Create saturation-highlighted preview
    saturated_mask = stacked >= 253  # True where pixels are clipped
    preview = cv2.cvtColor(stacked, cv2.COLOR_GRAY2BGR)
    preview[saturated_mask] = (0, 0, 255)  # mark clipped pixels in red

    preview_name = f"{wavelength}nm_{exposure_ms}ms_stacked_preview.jpg"
    preview_path = stack_dir / preview_name
    cv2.imwrite(str(preview_path), preview)

    return dng_path, preview_path



@app.route("/")
def index():
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    wavelength = int(request.form["wavelength"])
    exposure_ms = int(request.form["shutter"])
    frame_count = int(request.form["frames"])

    raw_dir = Path(request.form["raw_dir"])
    stack_base = Path(request.form["stack_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    stack_dir = stack_base / timestamp
    stack_dir.mkdir(parents=True, exist_ok=True)

    # Capture raw frames with metadata
    frames, controls = capture_images(wavelength, exposure_ms, frame_count, raw_dir)

    # Save stacked frame with metadata
    dng_path, preview_path = stack_images(frames, stack_dir, wavelength, exposure_ms, controls)

    rel_preview = preview_path.relative_to("static")
    rel_stack_dir = stack_dir.relative_to("static")

    return render_template(
        "result.html",
        result_image=str(rel_preview),
        session_dir=str(rel_stack_dir)
    )


@app.route("/download/<path:session_dir>")
def download(session_dir):
    abs_dir = BASE_DIR / session_dir
    zip_path = abs_dir.with_suffix(".zip")
    shutil.make_archive(str(abs_dir), 'zip', str(abs_dir))
    return send_file(zip_path, as_attachment=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
