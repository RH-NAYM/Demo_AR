from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import logging
import uvicorn




app = FastAPI()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# IP camera stream (from your camera app)
CAMERA_URL = "http://192.168.1.22:4747/video"

# Initialize VideoCapture with FFmpeg backend
cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

if not cap.isOpened():
    logger.error(f"Failed to open video stream: {CAMERA_URL}")
else:
    logger.info(f"Video stream opened successfully: {CAMERA_URL}")

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            logger.warning("Failed to grab frame from camera.")
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.warning("Failed to encode frame.")
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

@app.get("/")
def index():
    return {"message": "Go to /video to view the live stream."}

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")





if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5656)