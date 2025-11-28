import cv2
import time as t
import numpy as np

# HoughLinesP parameters
maxLineGap = 1
minLineLength = 1024
hough_threshold = 100

# Hough Circle parameters
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
    
    # Detect lines using HoughLinesP
    lines = cv2.HoughLinesP(
                gray_thresh, # Input edge image
                10, # Distance resolution in pixels
                np.pi/180, # Angle resolution in radians (FIXED: was 10)
                threshold=hough_threshold, # Min number of votes for valid line
                minLineLength=minLineLength, # Min allowed length of line
                maxLineGap=maxLineGap # Max allowed gap between line for joining them
                )

    result_frame = frame.copy()
    
    # Draw detected lines on the result frame (FIXED: was drawing on gray_thresh)
    if lines is not None:
        
        for points in lines:
            # Extracted points nested in the list
            x1, y1, x2, y2 = points[0]
            # Draw the lines on the result frame
            cv2.line(gray_thresh, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Detect circles
    current_dp = max(1, dp)
    hcircle = cv2.HoughCircles(gray_thresh, cv2.HOUGH_GRADIENT, current_dp, distance, 
                              param1=param1, param2=param2, 
                              minRadius=minradius, maxRadius=maxradius)

    # Draw detected circles
    if hcircle is not None:
        hcircle = np.round(hcircle[0, :]).astype("int")
        for (x, y, r) in hcircle:
            cv2.circle(gray_thresh, (x, y), r, (0, 0, 255), 4)
            cv2.circle(gray_thresh, (x, y), 2, (255, 255, 255), 3)
        
    # Display parameter values
    #cv2.putText(result_frame, f'Thresh: {thresh_value}', (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #cv2.putText(result_frame, f'MinLen: {minLineLength}', (10, 60), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    #cv2.putText(result_frame, f'MaxGap: {maxLineGap}', (10, 90), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return gray_thresh

# Parameter callback functions
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
    distance = max(1, val)  # Ensure at least 1

def maxLineGap_(val):
    global maxLineGap
    maxLineGap = val

def minLineLength_(val):
    global minLineLength
    minLineLength = val

def hough_threshold_(val):
    global hough_threshold
    hough_threshold = max(1, val)

cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera View with Lines')

# Hough Circle trackbars
cv2.createTrackbar('thresh_bg', 'Camera View with Lines', thresh_value, 255, thresh_bg)
cv2.createTrackbar('param1', 'Camera View with Lines', param1, 255, param1_)
cv2.createTrackbar('param2', 'Camera View with Lines', param2, 255, param2_)
cv2.createTrackbar('dp', 'Camera View with Lines', dp, 20, dp_) 
cv2.createTrackbar('minradius', 'Camera View with Lines', minradius, 500, minradius_)
cv2.createTrackbar('maxradius', 'Camera View with Lines', maxradius, 500, maxradius_)
cv2.createTrackbar('distance', 'Camera View with Lines', distance, 500, distance_)

# HoughLinesP trackbars
cv2.createTrackbar('maxLineGap', 'Camera View with Lines', maxLineGap, 100, maxLineGap_)
cv2.createTrackbar('minLineLength', 'Camera View with Lines', minLineLength, 2000, minLineLength_)
cv2.createTrackbar('hough_thresh', 'Camera View with Lines', hough_threshold, 300, hough_threshold_)

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
