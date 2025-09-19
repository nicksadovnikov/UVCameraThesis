import cv2
import numpy as np
import subprocess
from datetime import datetime

# 1. List formats (useful for debugging)
print("=== Supported formats ===")
try:
    output = subprocess.check_output(["v4l2-ctl", "--list-formats-ext", "-d", "/dev/video0"])
    print(output.decode())
except Exception as e:
    print("v4l2-ctl not available or failed:", e)

# 2. Open camera
cam = cv2.VideoCapture(0, cv2.CAP_V4L2)

if not cam.isOpened():
    raise RuntimeError("❌ Could not open /dev/video0")

# Try setting resolution (adjust to what camera supports)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cam.set(cv2.CAP_PROP_FPS, 10)

# 3. Capture N frames
N = 10
frames = []

print(f"Capturing {N} frames...")
for i in range(N):
    ret, frame = cam.read()
    if not ret:
        print(f"⚠️ Frame {i} failed")
        continue
    # Convert to 16-bit if needed
    frames.append(frame.astype(np.uint16))

cam.release()

# 4. Stack frames (mean)
stacked = np.mean(frames, axis=0).astype(np.uint16)

# 5. Save result
filename = f"stacked_{datetime.now():%Y%m%d_%H%M%S}.tiff"
cv2.imwrite(filename, stacked)
print(f"✅ Saved stacked image to {filename}")
