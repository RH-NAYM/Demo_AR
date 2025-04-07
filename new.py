import cv2
import numpy as np

img1 = cv2.imread('test/test.jpeg',0)
# img2 = cv2.imread('test/test2.jpg',0)

img2 = cv2.imread('test/test.jpeg',0)

orb = cv2.ORB_create(nfeatures=1000)


kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)



# imgKp1 = cv2.drawKeypoints(img1, kp1, None)
# imgKp2 = cv2.drawKeypoints(img2, kp2, None)


bf = cv2.BFMatcher()


matches = bf.knnMatch(des1, des2, k=2)

gd = []

for m, n in matches:
    if m.distance < 0.75 * n.distance:
        gd.append([m])
        


img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,gd,None,flags=2)


print(len(gd))

# cv2.imshow(kp1, imgKp1)
# cv2.imshow(kp2, imgKp2)

# cv2.imshow('Image_1 :',img1)
# cv2.imshow('Image_2 :',img2)
cv2.imshow('Image_3 :',img3)



cv2.waitKey(0)