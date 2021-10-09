'''
Source: 
https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
https://docs.opencv.org/3.4.15/da/d97/tutorial_threshold_inRange.html
https://stackoverflow.com/questions/47574173/how-can-i-draw-a-rectangle-around-a-colored-object-in-open-cv-python

'''

import cv2
import numpy as np
cap = cv2.VideoCapture(0)
def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)
while(1):
    _, frame = cap.read()
    frame = rescale_frame(frame, percent=40)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    lower_blue = np.array([115,200,100])
    upper_blue = np.array([125,255,150])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    lower_bound = np.array([90,0,0])
    upper_bound = np.array([255,60,60])
    mask = cv2.inRange(frame, lower_bound, upper_bound)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow('frame',frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
