import cv2

cam = cv2.VideoCapture(0, cv2.CAP_V4L2)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)

ret, frame = cam.read()
cam.release()

if not ret:
    print("❌ Failed to capture frame")
else:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("Frame shape:", gray.shape)
    print("Min pixel:", gray.min())
    print("Max pixel:", gray.max())
    cv2.imwrite("test_capture.jpg", gray)
    print("✅ Saved test_capture.jpg")
