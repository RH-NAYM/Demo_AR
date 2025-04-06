from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
import cv2
import threading
import time
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")

CAMERA_URL = "http://192.168.1.22:4747/video"
DISPLAY_NAME = "Hawk AI"  # <-- Your name here

cap = None
streaming = False
rotation_angle = 0  # 0, 90, 180, 270
flip_code = 1  # 0: Vertical flip, 1: Horizontal flip, -1: Both

lock = threading.Lock()

def start_capture():
    global cap
    cap = cv2.VideoCapture(CAMERA_URL, cv2.CAP_FFMPEG)

def stop_capture():
    global cap
    if cap:
        cap.release()
        cap = None

def overlay_name(frame, name):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    thickness = 2
    color = (0, 255, 0)  # Green
    margin = 10

    # Calculate text size to align it bottom-left
    text_size, _ = cv2.getTextSize(name, font, font_scale, thickness)
    text_x = margin
    text_y = frame.shape[0] - margin

    cv2.putText(frame, name, (text_x, text_y), font, font_scale, color, thickness, cv2.LINE_AA)
    return frame

def apply_transformations(frame):
    global rotation_angle, flip_code

    # Rotate the frame
    if rotation_angle != 0:
        (h, w) = frame.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        frame = cv2.warpAffine(frame, matrix, (w, h))

    # Flip the frame
    frame = cv2.flip(frame, flip_code)

    return frame

def generate_frames():
    global cap
    while streaming:
        if cap is None:
            time.sleep(0.1)
            continue
        success, frame = cap.read()
        if not success:
            break

        # Apply transformations and overlay name
        frame = apply_transformations(frame)
        frame = overlay_name(frame, DISPLAY_NAME)

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )
    stop_capture()

@app.get("/")
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/start")
def start_stream():
    global streaming
    with lock:
        if not streaming:
            streaming = True
            start_capture()
    return {"status": "stream started"}

@app.get("/stop")
def stop_stream():
    global streaming
    with lock:
        streaming = False
    return {"status": "stream stopped"}

@app.get("/rotate/{angle}")
def rotate_stream(angle: int):
    global rotation_angle
    with lock:
        rotation_angle = angle
    return {"status": f"rotation set to {angle} degrees"}

@app.get("/flip/{flip_type}")
def flip_stream(flip_type: int):
    global flip_code
    with lock:
        flip_code = flip_type
    return {"status": f"flip set to {flip_type}"}

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5656)
