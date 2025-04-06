# AR-application-in-python-OpenCV-
This project involves the development of an augmented reality application from scratch using OpenCV-python and NumPy.
The feature recognition is established using the oriented FAST algorithm. The algorithm detects keypoints and descriptors in the reference (target) image and the webcam feed image. The detected descriptors in the target are compared with the webcam feed using the Brute force method which uses the K nearest neighbor algorithm. Depending on the number of feature points matched with the reference the presence of the image in the scene is confirmed. Once the image is detected in the webcam feed homographic projection is performed. The RANSAC algorithm is used to eliminate any outliers. After that perspective transform is performed to find and corners of the image target in 3 dimensions and transform it into 2 dimensions, this is done to draw the bounding box around the image target in the scene. Finally, some binary image manipulation is done to warp the video in the image. 
