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

    cv.imshow("Original", newFrame)

    blur = cv.blur(newFrame, (5, 5))
    imageForAve = cv.accumulateWeighted(blur, imageForAve, .1)
    runningColorDepth = cv.convertScaleAbs(imageForAve, runningColorDepth)
    difference = cv.absdiff(newFrame, runningColorDepth, difference)
    gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    thresholdLow = cv.threshold(gray, 0, 5, cv.THRESH_BINARY)
    #gray_blur = cv.blur(thresholdLow, (5, 5))

    cv.imshow("test", thresholdLow)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()