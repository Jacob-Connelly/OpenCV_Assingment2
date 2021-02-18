import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
status, frame = cap.read()
width, height, channels = frame.shape
bw = np.zeros((width, height, 1), np.uint8)
imageForAve = np.zeros((width, height, 3), np.float32)
bright = 35 * np.ones((width, height, 3), np.uint8)
runningColorDepth = frame.copy()
difference = frame.copy()


while True:
    status, newFrame = cap.read()

    blur = cv.blur(newFrame, (5, 5))
    imageForAve = cv.accumulateWeighted(blur, imageForAve, .3)
    runningColorDepth = cv.convertScaleAbs(imageForAve, runningColorDepth)
    difference = cv.absdiff(newFrame, runningColorDepth, difference)
    gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    ret, thresholdLow = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    lowThresh_blur = cv.blur(thresholdLow, (5, 5))
    ret, thresholdHigh = cv.threshold(lowThresh_blur, 250, 255, cv.THRESH_BINARY)
    contours, hierarchy = cv.findContours(thresholdHigh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(newFrame, contours, -1, (0, 255, 0), 3)

    cv.imshow("test", thresholdHigh)
    cv.imshow("Original", newFrame)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()