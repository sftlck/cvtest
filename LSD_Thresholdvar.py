import cv2
import time as t

thresh_value = 220

def detect_lines(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    global thresh_value
    gray = cv2.threshold(gray, thresh_value, 255, cv2.THRESH_BINARY)[1]

    lsd = cv2.createLineSegmentDetector(refine=3, scale=None, sigma_scale=None, 
                                      quant=None, ang_th=None, log_eps=None, 
                                      density_th=None, n_bins=None)
    circle = cv2.circle(gray, (10,10), 50, (255,255,255))

    lines = lsd.detect(gray, width=0, prec=None, nfa=None)[0]
    if lines is not None:
        drawn_img = lsd.drawSegments(gray, lines)
    else:
        drawn_img = frame
    cv2.putText(drawn_img, f'Thresh: {thresh_value}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return drawn_img

def on_trackbar(val):
    global thresh_value
    thresh_value = val

cap = cv2.VideoCapture(0)
cv2.namedWindow('Camera View with Lines')

cv2.createTrackbar('Threshold', 'Camera View with Lines', thresh_value, 255, on_trackbar)

while True:
    ret, frame = cap.read()
    if not ret:
        print("noframecap")
        break
    t.sleep(0.02)
    frame_with_lines = detect_lines(frame)
    cv2.imshow('Camera View with Lines', frame_with_lines)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()