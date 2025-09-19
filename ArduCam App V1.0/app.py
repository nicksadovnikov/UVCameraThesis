import cv2
import numpy as np
from datetime import datetime

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
cam.set(cv2.CAP_PROP_FPS, 10)

frames = []
N = 10
for i in range(N):
    ret, frame = cam.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray.astype(np.uint16))

cam.release()

# Stack
stacked = np.mean(frames, axis=0).astype(np.uint16)

# Normalise so it's visible
stacked_norm = cv2.normalize(stacked, None, 0, 65535, cv2.NORM_MINMAX)

filename = f"stacked_{datetime.now():%Y%m%d_%H%M%S}.tiff"
cv2.imwrite(filename, stacked_norm)
print(f"✅ Saved stacked + normalized image to {filename}")
