import cv2
import time as t
import numpy as np

thresh_value = 133
param1 = 102
param2 = 131
dp = 4
minradius = 84
maxradius = 88
distance = 201

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    global thresh_value
    gray_thresh = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)[1]
    
    lsd = cv2.createLineSegmentDetector(0)
    
    current_dp = max(1, dp)
    hcircle = cv2.HoughCircles(gray_thresh, cv2.HOUGH_GRADIENT, current_dp, distance, 
                              param1=param1, param2=param2, 
                              minRadius=minradius, maxRadius=maxradius)

    result_frame = frame.copy()
    if hcircle is not None:
        hcircle = np.round(hcircle[0, :]).astype("int")
        for (x, y, r) in hcircle:
            cv2.circle(result_frame, (x, y), r, (0, 0, 255), 4)
            cv2.circle(result_frame, (x, y), 2, (255, 255, 255), 3)

    lines = lsd.detect(gray_thresh)[0]
    if lines is not None:
        drawn_img = lsd.drawSegments(result_frame, lines)
    else:
        drawn_img = result_frame
        
    cv2.putText(drawn_img, f'Thresh: {thresh_value}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return drawn_img

def thresh_bg(val):
    global thresh_value
    thresh_value = val

def param1_(val):
    global param1
    param1 = max(1, val)

def param2_(val):
    global param2
    param2 = max(1, val)

def dp_(val):
    global dp
    dp = max(1, val) 

def minradius_(val):
    global minradius
    minradius = val

def maxradius_(val):
    global maxradius
    maxradius = val

def distance_(val):
    global distance
    distance = val

cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera View with Lines')

cv2.createTrackbar('thresh_bg', 'Camera View with Lines', thresh_value, 255, thresh_bg)
cv2.createTrackbar('param1', 'Camera View with Lines', param1, 255, param1_)
cv2.createTrackbar('param2', 'Camera View with Lines', param2, 255, param2_)
cv2.createTrackbar('dp', 'Camera View with Lines', dp, 20, dp_) 
cv2.createTrackbar('minradius', 'Camera View with Lines', minradius, 500, minradius_)
cv2.createTrackbar('maxradius', 'Camera View with Lines', maxradius, 500, maxradius_)
cv2.createTrackbar('distance', 'Camera View with Lines', distance, 500, distance_)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break
        
    frame_with_lines = detect_lines(frame)
    cv2.imshow('Camera View with Lines', frame_with_lines)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
