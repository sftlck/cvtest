import cv2
import numpy as np

# Initial parameters
thresh_value = 178
param1 = 102
param2 = 131
dp = 1  # Inverse ratio of accumulator resolution
minDist = 20  # Minimum distance between detected centers
minRadius = 10  # Minimum radius
maxRadius = 100  # Maximum radius

def detect_lines(frame):
    # Create a copy of the frame to draw on
    result = frame.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold (optional - sometimes helps, sometimes not)
    if thresh_value > 0:
        _, thresholded = cv2.threshold(gray_blurred, thresh_value, 255, cv2.THRESH_BINARY)
    else:
        thresholded = gray_blurred
    
    # Debug: Show thresholded image
    
    thresholded = cv2.resize(gray_blurred, (500, 900))

    cv2.imshow('Thresholded', thresholded)
    
    
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(thresholded, 
                               cv2.HOUGH_GRADIENT, 
                               dp=dp,  # accumulator resolution
                               minDist=minDist,  # minimum distance between centers
                               param1=param1,  # upper threshold for Canny edge detector
                               param2=param2,  # threshold for center detection
                               minRadius=minRadius,
                               maxRadius=maxRadius)
    
    print(f"Detected circles: {circles is not None}")
    
    if circles is not None:
        circles = np.uint16(np.around(circles))
        print(f"Number of circles found: {len(circles[0])}")
        
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(result, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center of circle
            cv2.circle(result, (i[0], i[1]), 2, (0, 0, 255), 3)
            print(f"Circle center: ({i[0]}, {i[1]}), Radius: {i[2]}")
    
    return result

def thresh_trackbar(val):
    global thresh_value
    thresh_value = val

def minRadius_trackbar(val):
    global minRadius
    minRadius = max(1, val)  # Ensure minimum radius is at least 1

def maxRadius_trackbar(val):
    global maxRadius
    maxRadius = max(minRadius + 1, val)  # Ensure maxRadius > minRadius

def param1_trackbar(val):
    global param1
    param1 = max(1, val)  # Ensure at least 1

def param2_trackbar(val):
    global param2
    param2 = max(1, val)  # Ensure at least 1

def dp_trackbar(val):
    global dp
    dp = max(1, val) / 10.0  # dp is typically between 1 and 2

def minDist_trackbar(val):
    global minDist
    minDist = max(1, val)

# Create windows
cv2.namedWindow('Camera View with Lines')
cv2.namedWindow('Thresholded')

# Create trackbars
cv2.createTrackbar('Threshold', 'Camera View with Lines', thresh_value, 255, thresh_trackbar)
cv2.createTrackbar('minRadius', 'Camera View with Lines', minRadius, 200, minRadius_trackbar)
cv2.createTrackbar('maxRadius', 'Camera View with Lines', maxRadius, 200, maxRadius_trackbar)
cv2.createTrackbar('param1 (Canny)', 'Camera View with Lines', param1, 200, param1_trackbar)
cv2.createTrackbar('param2 (Accumulator)', 'Camera View with Lines', param2, 200, param2_trackbar)
cv2.createTrackbar('dp (Resolution)', 'Camera View with Lines', 10, 20, dp_trackbar)  # 1.0 to 2.0
cv2.createTrackbar('minDist', 'Camera View with Lines', minDist, 100, minDist_trackbar)

# Use webcam instead of static image for testing
cap = cv2.VideoCapture(0)

# Or for testing with your image:
# frame = cv2.imread(r'C:\Users\Castro\Desktop\Computa\TestesVC\12-08\imagem (4).jpg')

while True:

    frame = cv2.imread(r'C:\Users\Castro\Desktop\Computa\TestesVC\12-08\imagem (4).jpg')
    if frame is None:
        print("Failed to load image")
        break
    
    frame_with_circles = detect_lines(frame)
    
    frame_with_circles = cv2.resize(frame_with_circles, (500, 900))
    
    cv2.imshow('Camera View with Lines', frame_with_circles)
    
    print(f"Params: thresh={thresh_value}, minR={minRadius}, maxR={maxRadius}, param1={param1}, param2={param2}")
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()