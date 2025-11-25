import cv2
import numpy as np


camera_params = {
    'fx': 628.9513660368872,
    'fy': 631.2065961152636,
    'skew': 0.0,
    'cx': 320.2909594968069,
    'cy': 273.41095108720174,
    'width': 640,
    'height': 480,
    'r0': -356757302,
    'r1': 114489046
}

detector = cv2.QRCodeDetector()


def create_camera_matrix(params):
    return np.array([
        [params['fx'], params['skew'], params['cx']],
        [0, params['fy'], params['cy']],
        [0, 0, 1]
    ], dtype=np.float32)

def create_distortion_coefficients(params):
    scale_factor = 1e9
    k1 = params['r0'] / scale_factor
    k2 = params['r1'] / scale_factor
    
    return np.array([k1, k2, 0, 0, 0], dtype=np.float32)

def setup_undistort_maps(camera_matrix, dist_coeffs, width, height):
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )
    
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, 
        (width, height), cv2.CV_16SC2
    )
    
    return map1, map2, roi, new_camera_matrix

def main():
    camera_matrix = create_camera_matrix(camera_params)
    dist_coeffs = create_distortion_coefficients(camera_params)
    
    #print("Camera Matrix:")
    #print(camera_matrix)
    #print("\nDistortion Coefficients:")
    #print(dist_coeffs)
    
    map1, map2, roi, new_camera_matrix = setup_undistort_maps(
        camera_matrix, dist_coeffs, camera_params['width'], camera_params['height']
    )
    
    #print(f"\nROI for cropped image: {roi}")
    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_params['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_params['height'])
    
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    print("\nPress 'q' to quit, 's' to save current frame")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        _, img = cap.read()
        height, width = img.shape[:2]
        overlay = np.zeros((height, width, 3), dtype=np.uint8)
        data, bbox, _ = detector.detectAndDecode(img)
        
        if data:
            
            if bbox is not None:
                bbox = bbox[0].astype(int)
                cv2.polylines(img, 
                            [bbox], True, 
                            (0, 255, 0), 
                            2)
                cv2.polylines(overlay, 
                            [bbox], True, 
                            (255, 255, 255), 
                            2)
                
                colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]  
                for i, point in enumerate(bbox):

                    cv2.circle(overlay, 
                            tuple(point), 8, 
                            colors[i], -1)
                
                cv2.putText(overlay, 
                            data, 
                            (bbox[0][0], bbox[0][1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        (255, 255, 255), 
                        2)

        undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
        x, y, w, h = roi
        if w > 0 and h > 0:
            cropped = undistorted[y:y+h, x:x+w]
        else:
            cropped = undistorted
        
        original_resized = cv2.resize(frame, (640, 480))
        #corrected_resized = cv2.resize(cropped, (640, 480))
        
        combined = np.hstack([original_resized, 
                              overlay])
        
        #cv2.putText(combined, "Original", (10, 20), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        #cv2.putText(combined, "Corrected", (330, 20), 
        #           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('cam', combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == main():
    main()