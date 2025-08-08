from flask import Flask, render_template, Response
import subprocess
import cv2
import numpy as np

app = Flask(__name__)

def generate_frames():
    while True:
        # Capture one frame from libcamera
        result = subprocess.run(
            ["libcamera-jpeg", "-n", "-o", "-", "--width", "640", "--height", "480"],
            stdout=subprocess.PIPE
        )

        # Convert JPEG bytes to OpenCV frame
        jpg_data = result.stdout
        frame = cv2.imdecode(np.frombuffer(jpg_data, dtype=np.uint8), cv2.IMREAD_COLOR)

        # Re-encode to JPEG (in case modifications are needed)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield MJPEG-compatible chunk
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
