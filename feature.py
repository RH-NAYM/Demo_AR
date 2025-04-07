import cv2
import numpy as np
import os


path = 'test'
orb = cv2.ORB_create(nfeatures=1000)


images = []

classNames = []


myList = os.listdir(path)

print('Total Class: ', len(myList))


for cl in myList:
    imgCur = cv2.imread(f"{path}/{cl}",0)
    images.append(imgCur)
    
    classNames.append(os.path.splitext(cl)[0])
    
print(classNames)


def findDes(images):
    desList = []
    for img in images:
        kp,des = orb.detectAndCompute(img, None)
        desList.append(des)
        
    return desList
    


def findID(img, desList, thresh=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    
    bf = cv2.BFMatcher()

    matchList = []
    finalVal = -1
    
    try:    
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            gd = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    gd.append([m])
            print(len(gd))
            matchList.append(len(gd))
            
        print(matchList)
    except:
        pass
    
    
    
    if len(matchList) != 0:
        if max(matchList) > thresh:
            finalVal = matchList.index(max(matchList))
    return finalVal




desList = findDes(images)

print(len(desList))



cap = cv2.VideoCapture(0)


while True:
    success, img2 = cap.read()
    imgOrigin = img2.copy()
    
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    
    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOrigin, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        # cv2.rectangle(imgOrigin, (0, 0), (640, 100), (0, 255, 0), -1)
    
    
    
        
    cv2.imshow('original :',imgOrigin)
    # cv2.imshow('gray :',img2)



    cv2.waitKey(1)
    
    