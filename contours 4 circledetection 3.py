import cv2
import numpy as np

minLineLength = 0
maxLineGap = 0

canny1 = 184
canny2 = 418

thresh_value = 17
thresh_value2 = 9
param1 = 102
param2 = 3 * param1 
dp = 7
minradius = 167
maxradius = 127
distance = 201

history=100
varThreshold=120

gamma = 1

backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=True)

def detect_lines(frame):
    
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    medianblur = cv2.medianBlur(frame,3)
    frame = medianblur
    gray = cv2.cvtColor(medianblur, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, canny1, canny2)

    #cv2.imshow('Canny',gray)
    hcircle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, distance, param1=param1, param2=param2, minRadius=minradius, maxRadius=maxradius)
    
    circle_detected = False
    circle_info = None
    
    if hcircle is not None:
        hcircle = np.round(hcircle[0, :]).astype("int")
        
        for (x, y, r) in hcircle:
            circle_info = (x, y, r)
            circle_detected = True
            circle_tolerance = r

            cv2.circle(frame, (x, y), circle_tolerance, (0, 255, 0), 2)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), circle_tolerance, 255, -1)###MÁSCARA PARA ISOLAR ELEMENTOS DENTRO DO CÍRCULO
            
            frame_copy = frame.copy()
            fg_mask_full = backSub.apply(frame)
        
            fg_mask_circular = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)
            retval, mask_thresh = cv2.threshold(fg_mask_circular, 180, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            cv2.imshow('mask_eroded', mask_eroded)

            roi_mask = np.zeros_like(frame)
            cv2.circle(roi_mask, (x, y), r, (255, 255, 255), -1)
            frame_circle_only = cv2.bitwise_and(frame, roi_mask)
            frame_ct = frame.copy()
            cv2.drawContours(frame_ct, contours, -1, (0, 255, 0), 2) #### APLICA MÁSCARA PARA IDENTIFICAR COISAS DENTRO DO CÍRCULO
            cv2.imshow('Circle ROI_pre', frame_ct)
            frame_ct_circular = cv2.bitwise_and(frame_ct, roi_mask) ###APLICA MÁSCARA GRÁFICA PRETA DO LADO DE FORA DO CÍRCULO
            cv2.imshow('Circle ROI', frame_ct_circular)
            
            min_contour_area = 15
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

            for cnt in large_contours: #### ESSA REGIÃO AQUI CUIDA DOS RETÂNGULOS VERMELHOS
                x_rect, y_rect, w, h = cv2.boundingRect(cnt)
                
                contour_center = (x_rect + w//2, y_rect + h//2)
                dist_from_circle_center = np.sqrt((contour_center[0] - x)**2 + (contour_center[1] - y)**2)#### RAIO DO CÍRCULO IDENTIFICADO
                
                if dist_from_circle_center <= r:
                    cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 200), 1)
                    cv2.imshow('Control', frame)
            
            mask_thresh = cv2.Canny(mask_thresh, canny1, canny2)
            
            cv2.imshow('CANNY_LSD', mask_thresh)
            lines = cv2.HoughLinesP(mask_thresh, 1, np.pi/180, thresh_value, minLineLength=minLineLength, maxLineGap=maxLineGap)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    dist1 = np.sqrt((x1 - x)**2 + (y1 - y)**2)
                    dist2 = np.sqrt((x2 - x)**2 + (y2 - y)**2)
                    if dist1 <= r and dist2 <= r:
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                        
                        cv2.imshow('LSD', frame)

    
    return frame


def history_trackbar(val):
    global history
    history = val
    
def varThreshold_trackbar(val):
    global varThreshold
    varThreshold = val

def thresh_trackbar(val):
    global thresh_value
    thresh_value = val

def thresh2_trackbar(val):
    global thresh_value2
    thresh_value2 = val

def minLineLength_trackbar(val):
    global minLineLength
    minLineLength = val

def minradius_trackbar(val):
    global minradius
    minradius = val

def maxradius_trackbar(val):
    global maxradius
    maxradius = val

def canny1_trackbar(val):
    global canny1
    canny1 = val

def canny2_trackbar(val):
    global canny2
    canny2 = val

cap = cv2.VideoCapture(0)

cv2.namedWindow('Control')
cv2.createTrackbar('MinRadius', 'Control', minradius, 255, minradius_trackbar)
cv2.createTrackbar('MaxRadius', 'Control', maxradius, 255, maxradius_trackbar)
cv2.createTrackbar('Canny1', 'Control', minradius, 1500, canny1_trackbar)
cv2.createTrackbar('Canny2', 'Control', maxradius, 1500, canny2_trackbar)

cv2.createTrackbar('varThreshold', 'Control', varThreshold, 100, varThreshold_trackbar)
cv2.createTrackbar('history', 'Control', history, 1500, history_trackbar)

#cv2.namedWindow('Canny')
#cv2.createTrackbar('Canny1', 'Canny', canny1, 1000, canny1_trackbar)
#cv2.createTrackbar('Canny2', 'Canny', canny2, 1000, canny2_trackbar)

while True:
    ret, frame = cap.read()
    frame_with_lines = detect_lines(frame)
    #cv2.imshow("Final",frame_with_lines)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()