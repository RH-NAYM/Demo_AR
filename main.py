from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import cv2
import numpy as np
import base64
import uvicorn

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load target image and video
imgTarget = cv2.imread("static/markerimg.jpg")
video = cv2.VideoCapture("static/displayvideo1.mp4")
if imgTarget is None or not video.isOpened():
    raise RuntimeError("Missing resources.")

hT, wT, _ = imgTarget.shape
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)
frameCounter = 0

@app.get("/")
async def get():
    with open("static/index.html") as f:
        return HTMLResponse(content=f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected to WebSocket")

    global frameCounter
    while True:
        try:
            data = await websocket.receive_text()
            print("Frame received from client")

            img_data = base64.b64decode(data.split(",")[1])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                print("Could not decode frame")
                continue

            kp2, des2 = orb.detectAndCompute(frame, None)
            if des2 is None or des1 is None:
                await websocket.send_text("skip")
                continue

            matcher = cv2.BFMatcher()
            matches = matcher.knnMatch(des1, des2, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]

            if len(good) > 15:
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5)

                if matrix is not None:
                    if frameCounter >= video.get(cv2.CAP_PROP_FRAME_COUNT):
                        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frameCounter = 0

                    success, dispFrame = video.read()
                    frameCounter += 1

                    if success:
                        dispFrame = cv2.resize(dispFrame, (wT, hT))
                        warp = cv2.warpPerspective(dispFrame, matrix, (frame.shape[1], frame.shape[0]))

                        mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                        dst = cv2.perspectiveTransform(np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2), matrix)
                        cv2.fillPoly(mask, [np.int32(dst)], (255, 255, 255))
                        inv_mask = cv2.bitwise_not(mask)

                        frame_bg = cv2.bitwise_and(frame, frame, mask=inv_mask)
                        frame = cv2.bitwise_or(warp, frame_bg)

            _, jpeg = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(jpeg).decode("utf-8")
            await websocket.send_text(f"data:image/jpeg;base64,{b64}")

        except Exception as e:
            print("WebSocket connection closed or error:", e)
            break

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)

