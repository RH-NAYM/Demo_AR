from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import uvicorn

app = FastAPI()

CAMERA_URL = "http://192.168.1.22:4747/video"

cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)


def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
        
            break
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
        
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