import cv2
import numpy as np

maxCorners = 104
qualityLevel = 12
minDistance = 12
blockSize = 23

def nothing(x):
    pass

img = cv2.imread(r'C:\Users\AdmPGE\Desktop\Castro\Viscomp\Imagens\teste12.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('Good Features to Track')
cv2.createTrackbar('Max Corners', 'Good Features to Track', maxCorners, 500, nothing)
cv2.createTrackbar('Quality Level (x100)', 'Good Features to Track', qualityLevel, 100, nothing)
cv2.createTrackbar('Min Distance', 'Good Features to Track', minDistance, 50, nothing)
cv2.createTrackbar('Block Size', 'Good Features to Track', blockSize, 31, nothing)

while True:
    maxCorners = cv2.getTrackbarPos('Max Corners', 'Good Features to Track')
    qualityLevel_val = cv2.getTrackbarPos('Quality Level (x100)', 'Good Features to Track')
    minDistance = cv2.getTrackbarPos('Min Distance', 'Good Features to Track')
    blockSize = cv2.getTrackbarPos('Block Size', 'Good Features to Track')
    
    qualityLevel = qualityLevel_val / 100.0
    
    if maxCorners < 1:
        maxCorners = 1
    
    if minDistance < 1:
        minDistance = 1
    
    if blockSize < 3:
        blockSize = 3
    if blockSize % 2 == 0:  # Make sure it's odd
        blockSize += 1
    
    img_copy = img.copy()
    
    feature_params = dict(
        maxCorners=maxCorners,
        qualityLevel=qualityLevel,
        minDistance=minDistance,
        blockSize=blockSize
    )
    
    corners = cv2.goodFeaturesToTrack(gray, **feature_params)
    
    corners_detected = 0
    if corners is not None:
        corners = np.intp(corners)
        corners_detected = len(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img_copy, (x, y), 3, (0, 0, 255), 1)
    
    #cv2.putText(img_copy, f'Max Corners: {maxCorners}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(img_copy, f'Quality Level: {qualityLevel:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(img_copy, f'Min Distance: {minDistance}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(img_copy, f'Block Size: {blockSize}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(img_copy, f'Corners Found: {corners_detected}', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('Good Features to Track', img_copy)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()

if corners is not None:
    cv2.imshow('Final Result', img_copy)
    print(f"Final parameters used:")
    print(f"  Max Corners: {maxCorners}")
    print(f"  Quality Level: {qualityLevel:.2f}")
    print(f"  Min Distance: {minDistance}")
    print(f"  Block Size: {blockSize}")
    print(f"  Corners detected: {corners_detected}")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No features were detected in the image. Try adjusting parameters or image source.")