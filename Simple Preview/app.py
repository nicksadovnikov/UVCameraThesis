from flask import Flask, Response, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    # Launch libcamera-vid as a subprocess and stream MJPEG
    cmd = [
        'libcamera-vid',
        '-t', '0',                    # unlimited time
        '--inline',                  # required for MJPEG streaming
        '--width', '640',
        '--height', '480',
        '--codec', 'mjpeg',
        '-o', '-'                    # output to stdout
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

    def generate():
        try:
            while True:
                chunk = process.stdout.read(1024)
                if not chunk:
                    break
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + chunk + b'\r\n')
        finally:
            process.terminate()

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

