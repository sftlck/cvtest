import cv2
import time as t
import numpy as np

thresh_value = 0
minLineLength = 0
maxLineGap = 0

canny1 = 0
canny2 = 139

thresh_value = 0
param1 = 102
param2 = 131
dp = 7
minradius = 0
maxradius = 0
distance = 201

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    gray = cv2.blur(gray, (2, 2))
    
    gray = cv2.Canny(gray, canny1, canny2)
    
    gray = cv2.blur(gray, (2, 2))
    lines = cv2.HoughLinesP(gray, 1, np.pi/180, thresh_value, minLineLength=minLineLength, maxLineGap=maxLineGap)
    
    gray_thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)[1]
 
    hcircle = cv2.HoughCircles(gray_thresh, cv2.HOUGH_GRADIENT, 7, 139, 
                              param1=139, param2=240, 
                              minRadius=132, maxRadius=5)

    if hcircle is not None:
        hcircle = np.round(hcircle[0, :]).astype("int")
        for (x, y, r) in hcircle:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 3)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 2)
            
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dist1 = np.sqrt((x1 - x)**2 + (y1 - y)**2)
                    dist2 = np.sqrt((x2 - x)**2 + (y2 - y)**2)
                    if dist1 <= r and dist2 <= r:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    
    return frame

def thresh_trackbar(val):
    global param2
    param2 = val

def minLineLength_trackbar(val):
    global param1
    param1 = val

def maxLineGap_trackbar(val):
    global minradius
    minradius = val

def canny1_trackbar(val):
    global maxradius
    maxradius = val

def canny2_trackbar(val):
    global distance
    distance = val


cap = cv2.VideoCapture(0)

cv2.namedWindow('Camera View with Lines')
cv2.createTrackbar('Threshold', 'Camera View with Lines', param2, 255, thresh_trackbar)
cv2.createTrackbar('minlinelength', 'Camera View with Lines', param1, 255, minLineLength_trackbar)
cv2.createTrackbar('maxlinegap', 'Camera View with Lines', minradius, 255, maxLineGap_trackbar)
cv2.createTrackbar('canny1', 'Camera View with Lines', maxradius, 255, canny1_trackbar)
cv2.createTrackbar('canny2', 'Camera View with Lines', distance, 255, canny2_trackbar)

while True:
    ret, frame = cap.read()
    frame_with_lines = detect_lines(frame)
    
    cv2.imshow('Camera View with Lines', frame_with_lines)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
