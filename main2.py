from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from starlette.requests import Request
import uvicorn
import cv2
import threading
import tkinter as tk
from tkinter import Toplevel
from PIL import Image, ImageTk

app = FastAPI()

# Set up templates and static files (for JS/CSS)
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Create a Tkinter window to display the camera feed
def start_camera_window():
    # Initialize OpenCV video capture (camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access the camera.")
        return

    # Create a Tkinter window
    window = Toplevel()
    window.title("Camera Feed")
    window.geometry("640x480")

    # Label to display video stream
    label = tk.Label(window)
    label.pack()

    def update_frame():
        ret, frame = cap.read()
        if ret:
            # Convert the frame to RGB (Tkinter requires RGB format)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img_tk = ImageTk.PhotoImage(image=img)
            label.img_tk = img_tk  # Store reference to avoid garbage collection
            label.config(image=img_tk)
        window.after(10, update_frame)  # Update every 10 ms

    update_frame()  # Start updating the frames

    window.mainloop()  # Start the Tkinter event loop

# Serve the main HTML page with the camera stream
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# WebSocket for communication with the front-end for camera stream
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received data: {data}")
            # Send back detected object information or other data
            await websocket.send_text(f"Detected: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")

# FastAPI endpoint to trigger the Tkinter camera window
@app.get("/start_camera")
async def start_camera():
    # Run the camera window in a separate thread to not block the FastAPI server
    threading.Thread(target=start_camera_window).start()
    return {"message": "Camera window started"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5656)
