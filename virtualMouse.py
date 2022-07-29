import cv2
import autopy
from pynput.mouse import Button, Controller
import handTracker
import numpy as np
import time




cap = cv2.VideoCapture(0)

cameraWidth, cameraHeight = 640, 450
cap.set(3, cameraWidth)
cap.set(4, cameraHeight)
screenWidth, screenHeight = autopy.screen.size()
#print(screenWidth, screenHeight)

previousTime, currentTime = 0, 0
prevLocX, prevLocY = 0, 0
currLocX, currLocY = 0, 0

frameReduction = 100
smootheningValue = 7

mouse = Controller()
detector = handTracker.HandTracker(maxHands=1)


while True:

    success, img = cap.read()
    img = detector.findHands(img)
    landmarks, boundingBox = detector.findPosition(img)
    if len(landmarks) != 0:
        x1, y1 = landmarks[8][1:]
        x2, y2 = landmarks[12][1:]
        #print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        #print(fingers)
        cv2.rectangle(img, (frameReduction, frameReduction), (cameraWidth-frameReduction, cameraHeight-frameReduction), (255, 0, 255))
        if fingers[1]==1 and fingers[2]==0:
            x3 = np.interp(x1, (frameReduction, cameraWidth-frameReduction), (0, screenWidth))
            y3 = np.interp(y1, (frameReduction, cameraHeight-frameReduction), (0, screenHeight))
            currLocX = prevLocX+(x3 - prevLocX)/ smootheningValue
            currLocY = prevLocY+(y3 - prevLocY)/ smootheningValue

            autopy.mouse.move(screenWidth - currLocX, currLocY)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), cv2.FILLED)
            prevLocX, prevLocY = currLocX, currLocY

        #left clicking
        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(8, 12, img)
            #print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)
                autopy.mouse.click()

        #right clicking
        if fingers[1]==1 and fingers[2]==1:
            length, img, lineInfo = detector.findDistance(12, 16, img)
            #print(length)
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0, 255, 255), cv2.FILLED)
                mouse.click(Button.right, 1)


    currentTime = time.time()
    fps = 1/(currentTime - previousTime)
    previousTime = currentTime
    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key==27:
        break




cap.release()
cv2.destroyAllWindows()
