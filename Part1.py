import cv2 as cv
import numpy as np

# Global variables to hold high and low values
global hH
global hL
global sH
global sL
global vH
global vL


# function used in mouse callback to print out HSV values
def print_HSV(event, x, y, flags, param):
    global mouseX, mouseY
    if event == cv.EVENT_LBUTTONDBLCLK:
        cv.circle(img, (x, y), 100, (255, 0, 0), -1)
        mouseX, mouseY = x, y
        print("HSV Values: ")
        print(hsv[mouseY][mouseX])


# do nothing function to pass in when making the trackbars
def nothing(x):
    pass


# starts video capture and creates 3 windows
cap = cv.VideoCapture(0)
cv.namedWindow("HSV Video")
cv.namedWindow("Track Bar")
cv.namedWindow("Tracker")
cv.setMouseCallback('HSV Video', print_HSV)

# Create track bars for HSV value ranges
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

    # shows hsv video
    cv.imshow("HSV Video", hsv)

    k = cv.waitKey(1)
    if k == 27:
        break

    # creates track bars and sets values to global high and lows for each HSV
    hH = cv.getTrackbarPos("Hue High", "Track Bar")
    hL = cv.getTrackbarPos("Hue Low", "Track Bar")
    sH = cv.getTrackbarPos("Saturation High", "Track Bar")
    sL = cv.getTrackbarPos("Saturation Low", "Track Bar")
    vH = cv.getTrackbarPos("Value High", "Track Bar")
    vL = cv.getTrackbarPos("Value Low", "Track Bar")

    # creates numpy arrays for low and high values
    max_HSV = np.array([hH, sH, vH])
    low_HSV = np.array([hL, sL, vL])

    # thresholds low and high values then uses a mask to erode and dilate the black and white image
    frame_threshold = cv.inRange(hsv, low_HSV, max_HSV)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(frame_threshold, kernel, iterations=1)
    dilation = cv.dilate(erosion, kernel, iterations=1)

    # shows tracked object from HSV trackbar values after being eroded and dilated
    cv.imshow("Tracker", dilation)

cv.destroyAllWindows()
