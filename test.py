import cv2
import numpy as np
cap = cv2.VideoCapture(0)
imgTarget = cv2.imread('markerimg.jpg')
video = cv2.VideoCapture('displayvideo1.mp4')

if imgTarget is None:
    raise FileNotFoundError("Target image not found.")
if not video.isOpened():
    raise FileNotFoundError("Video file not found or can't be opened.")
if not cap.isOpened():
    raise RuntimeError("Webcam could not be accessed.")
success, dispVideo = video.read()
if not success:
    raise RuntimeError("Couldn't read from the video.")
hT, wT, cT = imgTarget.shape
dispVideoResized = cv2.resize(dispVideo, (wT, hT))
orb = cv2.ORB_create(nfeatures=1000)
kp1, des1 = orb.detectAndCompute(imgTarget, None)

detection = False
frameCounter = 0

while True:
    success, WebcamFeed = cap.read()
    if not success:
        print("Failed to grab frame from webcam.")
        break

    AugmentedImage = WebcamFeed.copy()
    kp2, des2 = orb.detectAndCompute(WebcamFeed, None)

    if des2 is None or des1 is None:
        cv2.imshow('AugmentedImage', AugmentedImage)
        cv2.waitKey(1)
        continue

    bruteForce = cv2.BFMatcher()
    matches = bruteForce.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    print(f"Good matches: {len(good)}")
    imgFeatures = cv2.drawMatches(imgTarget, kp1, WebcamFeed, kp2, good, None, flags=2)

    if len(good) > 15:
        detection = True
        srcpt = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        despt = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(srcpt, despt, cv2.RANSAC, 5)
        if matrix is not None:
            pts = np.float32([[0, 0], [0, hT], [wT, hT], [wT, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            img2 = cv2.polylines(WebcamFeed, [np.int32(dst)], True, (255, 0, 255), 3)

            if frameCounter >= video.get(cv2.CAP_PROP_FRAME_COUNT):
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frameCounter = 0

            success, dispVideo = video.read()
            if success:
                dispVideoResized = cv2.resize(dispVideo, (wT, hT))
                imgWarp = cv2.warpPerspective(dispVideoResized, matrix, (WebcamFeed.shape[1], WebcamFeed.shape[0]))

                maskNew = np.zeros((WebcamFeed.shape[0], WebcamFeed.shape[1]), np.uint8)
                cv2.fillPoly(maskNew, [np.int32(dst)], (255, 255, 255))
                mskInv = cv2.bitwise_not(maskNew)

                AugmentedImage = cv2.bitwise_and(AugmentedImage, AugmentedImage, mask=mskInv)
                AugmentedImage = cv2.bitwise_or(imgWarp, AugmentedImage)

            frameCounter += 1
        else:
            print("Homography matrix could not be computed.")
            detection = False
    else:
        detection = False
        video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frameCounter = 0

    cv2.imshow('imgFeatures',imgFeatures)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video.release()
cv2.destroyAllWindows()
