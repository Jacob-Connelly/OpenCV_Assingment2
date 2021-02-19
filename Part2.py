import cv2 as cv
import numpy as np

# starts video capture and creates blank variables to be used from step 2
cap = cv.VideoCapture(0)
status, frame = cap.read()
width, height, channels = frame.shape
bw = np.zeros((width, height, 1), np.uint8)
imageForAve = np.zeros((width, height, 3), np.float32)
runningColorDepth = frame.copy()
difference = frame.copy()

print("start")
while True:
    status, newFrame = cap.read()

    # blurs image then take running average and converts to make same bit as newFrame
    blur = cv.blur(newFrame, (5, 5))
    imageForAve = cv.accumulateWeighted(blur, imageForAve, .4)
    runningColorDepth = cv.convertScaleAbs(imageForAve, runningColorDepth)

    # Take difference of the new frame and running average then turn to grayscale
    difference = cv.absdiff(newFrame, runningColorDepth, difference)
    gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)

    # threshold with a low value, blur then threshold with a high value
    ret, thresholdLow = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)
    lowThresh_blur = cv.blur(thresholdLow, (5, 5))
    ret, thresholdHigh = cv.threshold(lowThresh_blur, 250, 255, cv.THRESH_BINARY)

    # finds contours then thresholds which contours to take, drawing rectangle over contours within threshold
    contours, hierarchy = cv.findContours(thresholdHigh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        (x, y, w, h) = cv.boundingRect(contour)

        if cv.contourArea(contour) < 5000:
            continue
        cv.rectangle(newFrame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # shows black and white movement image and contour image that draws rectangles where motion is detected
    cv.imshow("Black and White", thresholdHigh)
    cv.imshow("Contours", newFrame)

    k = cv.waitKey(1)
    if k == 27:
        break

cv.destroyAllWindows()
