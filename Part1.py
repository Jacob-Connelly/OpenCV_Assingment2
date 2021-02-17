import cv2 as cv
import numpy as np


global hH
global hL
global sH
global sL
global vH
global vL


def print_HSV(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        print("HSV Values: ")
        print(hsv[mouseY][mouseX])


def nothing(x):
    pass


cap = cv.VideoCapture(0)
cv.namedWindow("HSV Video")
cv.namedWindow("Track Bar")
cv.namedWindow("Tracker")
cv.setMouseCallback('HSV Video', print_HSV)

cv.createTrackbar("Hue High", "Track Bar", 0, 179, nothing)
cv.createTrackbar("Hue Low", "Track Bar", 0, 179, nothing)
cv.createTrackbar("Saturation High", "Track Bar", 0, 255, nothing)
cv.createTrackbar("Saturation Low", "Track Bar", 0, 255, nothing)
cv.createTrackbar("Value High", "Track Bar", 0, 255, nothing)
cv.createTrackbar("Value Low", "Track Bar", 0, 255, nothing)

print("start")
while True:
    status, img = cap.read()

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    cv.imshow("HSV Video", hsv)

    k = cv.waitKey(1)
    if k == 27:
        break

    hH = cv.getTrackbarPos("Hue High", "Track Bar")
    hL = cv.getTrackbarPos("Hue Low", "Track Bar")
    sH = cv.getTrackbarPos("Saturation High", "Track Bar")
    sL = cv.getTrackbarPos("Saturation Low", "Track Bar")
    vH = cv.getTrackbarPos("Value High", "Track Bar")
    vL = cv.getTrackbarPos("Value Low", "Track Bar")

    max_HSV = np.array([hH, sH, vH])
    low_HSV = np.array([hL, sL, vL])

    frame_threshold = cv.inRange(hsv, low_HSV, max_HSV)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(frame_threshold, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=1)
    cv.imshow("Tracker", dilation)

cv.destroyAllWindows()
