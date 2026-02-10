import cv2
import numpy as np
import math

############# FUNCIONAL PARA MOSTRADORES ANALÓGICOS DE ESCALA COM NÚMEROS NATURAIS

const1 = 0.9

override_height = 1080
override_width = 1920

minLineLength = 0
maxLineGap = 0

canny1 = 10
canny2 = 255

############## TEMPORÁRIO
thresh_value = 213
#ORIGINAL thresh_value = 214
thresh_value2 = 255
param1 = 102
param2 = 3 * param1 
dp = 7

######################################### TWEAK IN ORDER TO FIND AN ACCEPTABLE RADIUS
minradius = 390
maxradius = minradius + 10
#########################################
distance = 201

######## history era 50
history=100
varThreshold=120

gamma = 1

alpha = 1.5
beta = 0    

min_contour_area = 100

#### GLOBAL VALORES PARA REGISTRAR LINHA
global_circle_center = None
global_extended_point = None
global_black_foreground = None

#### GLOBAL VALORES PARA CALCULAR POSIÇÃO ATUAL
global_start_position_c_value = None
global_final_position_c_value = None
global_last_registered_line = None

#### GLOBAL VALORES PARA ÂNGULO E CONVERSÃO DE UNIDADE DE MEDIDA
global_v_inicial = None
global_v_final = None
range_value = 35

#### GLOBAL TESTE DE RESOLUÇÃO DO MOSTRADOR
res_inst = 0.1
value_range = 35
res_calc = str(res_inst); res_calc = res_calc.find("1") - 1

results = []
global_results = []
angles = []
list_positions = []
start_position = []

frame_counter = 0

backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=True)

####### ROI SELECTION GLOBALS #######
roi_selected = False
roi_circle = None 
roi_selection_active = False
selection_window = None

def display_last_value():
    if len(results) > 0:
        return results[-1]
    else:
        return []

def display_current_angle():
    global global_last_registered_line, global_circle_center, global_extended_point, list_positions
    current_angle_display = 0
    
    if global_last_registered_line is not None and global_circle_center is not None and global_extended_point is not None:
        if len(list_positions) > 0:
            last_line = list_positions[-1]
            current_line = (global_circle_center, global_extended_point)
            current_angle_display = calculate_smallest_angle_between_lines(last_line, current_line)
        else:
            current_angle_display = 0
    
    return current_angle_display

def needle_angle(center, tip_point):
    cx, cy = center
    tx, ty = tip_point
    
    angle_rad = math.atan2(ty - cy, tx - cx)
    current_angle_deg_ref = math.degrees(angle_rad)
    
    if current_angle_deg_ref < 0:
        current_angle_deg_ref += 360
    
    return current_angle_deg_ref

def line_circle_intersection(line_point1, line_point2, circle_center, circle_radius):
    x1, y1 = line_point1
    x2,y2 = line_point2
    cx,cy =circle_center
    r= circle_radius
    dx=x2 - x1
    dy= y2 - y1
    
    fx= x1 - cx
    fy= y1 - cy
    
    a = dx*dx + dy*dy
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    
    discriminant = b*b - 4*a*c
    
    if discriminant < 0:
        return []
    
    t1 = (-b + np.sqrt(discriminant)) / (2*a)
    t2 = (-b - np.sqrt(discriminant)) / (2*a)
    
    intersections = []
    
    if 0 <= t1 <= 1 or not (t1 < 0 or t1 > 1):
        ix1 = x1 + t1 * dx
        iy1 = y1 + t1 * dy
        intersections.append((int(ix1), int(iy1)))
    
    if 0 <= t2 <= 1 or not (t2 < 0 or t2 > 1):
        ix2 = x1 + t2 * dx
        iy2 = y1 + t2 * dy
        intersections.append((int(ix2), int(iy2)))
    
    return intersections

class ROISelection:
    def __init__(self):
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.circle = None 
        self.roi_confirmed = False
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.start_point = (x, y)
            self.drawing = True
            self.circle = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            self.end_point = (x, y)

            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                radius = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
                self.circle = (center_x, center_y, radius)
    
    def draw_circle(self, frame):
        if self.drawing and self.start_point and self.end_point:
            x1, y1 = self.start_point
            x2, y2 = self.end_point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = int(math.sqrt((x2 - x1)**2 + (y2 - y1)**2) / 2)
            
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 1)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 1)
            
        elif self.circle and not self.roi_confirmed:
            center_x, center_y, radius = self.circle
            cv2.circle(frame, (center_x, center_y), radius, (0, 255, 0), 1)
            cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), 1)
            
        return frame

def select_roi_window():
    global roi_selected, roi_circle, roi_selection_active

    
    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, override_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, override_height)
    
    if not cap.isOpened():
        print(">>> ERROR CAMERA INDEX NOT FOUND")
        return
    
    roi_selector = ROISelection()

    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", roi_selector.mouse_callback)

    cv2.namedWindow("ROI Controls")
    cv2.createTrackbar("Adjust Radius", "ROI Controls", 300, 500, lambda x: None)
    
    while True:

        ret, frame = cap.read()
        if not ret:
            break
        
        display_frame = frame.copy()
    
        display_frame = roi_selector.draw_circle(display_frame)
        
        cv2.putText(display_frame, "select roi",                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(display_frame, "c/r",                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        cv2.putText(display_frame, "q",                         (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1)
        
        if roi_selector.circle:
            x, y, r = roi_selector.circle
            cv2.putText(display_frame, f"x,y: ({x}, {y})",      (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display_frame, f"r: {r}",               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Select ROI", display_frame)
        
        if roi_selector.circle:
            radius_adjust = cv2.getTrackbarPos("Adjust Radius", "ROI Controls")
            x, y, _ = roi_selector.circle
            roi_selector.circle = (x, y, radius_adjust)
        
        key = cv2.waitKey(1) & 0xFF
    
        if key == 32:
            if roi_selector.circle:
                roi_selector.roi_confirmed = True
                roi_circle = roi_selector.circle
                roi_selected = True
                print(f">>> ROI SELECTED")
                #print(f"ROI confirmed: Center({roi_circle[0]}, {roi_circle[1]}), Radius: {roi_circle[2]}")
                
                inner_r = int(roi_circle[2] * const1)
                outer_r = roi_circle[2]
                hh, ww = display_frame.shape[:2]
                
                white_background = np.ones((hh, ww, 3), dtype=np.uint8) * 255
    
                mask_ring = np.zeros((hh, ww), dtype=np.uint8)
                cv2.circle(mask_ring, (x, y), outer_r, 255, -1)
                cv2.circle(mask_ring, (x, y), inner_r, 0, -1)

                #ring_from_original = cv2.bitwise_and(display_frame, display_frame, mask=mask_ring)
                #
                #cv2.imshow('ring_from_original',ring_from_original)
#
                #mask_bg = cv2.bitwise_not(mask_ring)
                #white_bg = cv2.bitwise_and(white_background, white_background, mask=mask_bg)
#
                #final_image = cv2.add(ring_from_original, white_bg)
#
                #x1 = max(0, x - outer_r)
                #y1 = max(0, y - outer_r)
                #x2 = min(ww, x + outer_r)
                #y2 = min(hh, y + outer_r)
#
                #cropped_image = final_image[y1:y2, x1:x2]
#
                #cv2.imshow('cropped_image',cropped_image)
#
                #output_gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
#
                #cv2.imshow('output_gray',output_gray)
                #mask, output_thresh = cv2.threshold(output_gray, thresh_value, thresh_value2, cv2.THRESH_BINARY)
                #cv2.imshow('output_thresh',output_thresh)
                

                break
            else:
                print(">>> INSERT CIRCLE TO DETECT")
                
        elif key == ord('r'):
            roi_selector.start_point = None
            roi_selector.end_point = None
            roi_selector.circle = None
            roi_selector.drawing = False
            print(">>> ROI RESET")
            
        elif key == ord('q'):
            print(">>> ROI SELECTION CANCELED")
            break
    
    cap.release()
    cv2.destroyWindow("Select ROI")
    cv2.destroyWindow("ROI Controls")
    
    if not roi_selected:
        print("Warning: No ROI selected. Using HoughCircles detection.")
    else:
        print(f"(x,y,r) = {roi_circle}")

def thresh_trackbar(val):
    global thresh_value
    thresh_value = val

def thresh2_trackbar(val):
    global thresh_value2
    thresh_value2 = val

#cv2.namedWindow('output_thresh')
#cv2.createTrackbar('thresh_value', 'output_thresh', thresh_value, 255, thresh_trackbar)
#cv2.createTrackbar('thresh_value2', 'output_thresh', thresh_value2, 255, thresh2_trackbar)

def read(frame):
    global global_black_foreground, global_circle_center, global_extended_point
    global roi_selected, roi_circle
    
    black_foreground = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cropped_frame = frame.copy()
    
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    #cv2.imshow('frame', frame)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, canny1, canny2)

    line_coordinates_list = []
    all_points_for_fitting = []
    
    circle_detected = False
    circle_data = None
    
    if roi_selected and roi_circle:
        x, y, r = roi_circle
        circle_data = (x, y, r)
        circle_detected = True
        circle_tolerance = int(circle_data[2] * 0.9)
    else:
        hcircle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, distance, param1=param1, param2=param2, minRadius=minradius, maxRadius=maxradius)
        
        if hcircle is not None:
            hcircle = np.round(hcircle[0, :]).astype("int")
            
            for (x, y, r) in hcircle:
                circle_data = (x, y, r * 0.9) 
                circle_detected = True
                circle_tolerance = int(circle_data[2] * 0.9)
    
    if circle_detected and circle_data:
        x, y, r = circle_data[0], circle_data[1], circle_data[2]
        circle_tolerance = int(r * 0.9)
        
        fg_mask_full = backSub.apply(frame)

        cv2.circle(frame,               (x, y), circle_tolerance,   (0, 255, 0),     1)
        cv2.circle(frame,               (x, y), 2,                  (0, 255, 0),     1)
        cv2.circle(black_foreground,    (x, y), circle_tolerance,   (0, 255, 0),     1)
        cv2.circle(black_foreground,    (x, y), 2,                  (0, 255, 0),     1)
        cv2.circle(mask,                (x, y), circle_tolerance,   255,            -1)
        
        fg_mask_circular = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)
        retval, mask_thresh = cv2.threshold(fg_mask_circular, thresh_value, thresh_value2, cv2.THRESH_BINARY)
        
        cv2.imshow('mask_thresh', mask_thresh)
        cv2.imshow('fg_mask_circular', fg_mask_circular)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        frame_ct = frame.copy()
        
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        rectangle_mask = np.zeros_like(mask_thresh)
        
        for cnt in large_contours:
            x_rect, y_rect, w, h = cv2.boundingRect(cnt)
            contour_center = (x_rect + w//2, y_rect + h//2)
            dist_from_circle_center = np.sqrt((contour_center[0] - x)**2 + (contour_center[1] - y)**2)
            
            if dist_from_circle_center <= r:
                cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 255), 1)
                cv2.rectangle(rectangle_mask, (x_rect, y_rect), (x_rect + w, y_rect + h), (255, 255, 255), -1)
                #cv2.imshow('Control', 0)
                
        masked_canny = cv2.bitwise_and(mask_thresh, rectangle_mask)
        cv2.imshow('CANNY_LSD', masked_canny)

        indices = np.where(masked_canny > 0)
        masked_canny_points = list(zip(indices[1], indices[0]))

        circle_points_for_fitting = []

        for px, py in masked_canny_points:
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            
            if (py < rectangle_mask.shape[0] and px < rectangle_mask.shape[1] and 
                rectangle_mask[py, px] > 0 and dist <= r):
                circle_points_for_fitting.append([px, py])
                all_points_for_fitting.extend(circle_points_for_fitting)

    needle_origin = None
    needle_end = None
    
    if len(all_points_for_fitting) >= 2:
        fitline = np.array(all_points_for_fitting, dtype=np.float32)
        
        [vx_all, vy_all, x0_all, y0_all] = cv2.fitLine(fitline, cv2.DIST_L2, 0, .01, .01)
        
        if abs(vx_all[0]) > 1e-6:
            
            needle_origin = (circle_data[0], circle_data[1])
            
            height, width = frame.shape[:2]
            needle_tip_x = int(circle_data[0] + vx_all[0] * circle_data[2])
            needle_tip_y = int(circle_data[1] + vy_all[0] * circle_data[2])
            needle_end = (needle_tip_x, needle_tip_y)
            
            slope_all = vy_all[0] / vx_all[0]  
            left_y_all = int(y0_all[0] + slope_all * (0 - x0_all[0]))
            right_y_all = int(y0_all[0] + slope_all * (width - 1 - x0_all[0]))
            right_all = (width - 1, right_y_all)
            left_all = (0, left_y_all)
            
            cv2.line(frame, left_all, right_all, (0, 255, 255), 1)
            
            centroid = (int(x0_all), int(y0_all))
            cv2.circle(frame, centroid, 5, (0, 255, 0), 1)
            cv2.circle(black_foreground, centroid, 5, (0, 255, 0), 1)

            if circle_detected and circle_data and needle_origin and needle_end:
                circle_center = (circle_data[0], circle_data[1])
                circle_radius = circle_data[2]
                
                dir_x = needle_end[0] - needle_origin[0]
                dir_y = needle_end[1] - needle_origin[1]
                
                length = np.sqrt(dir_x*dir_x + dir_y*dir_y)
                if length > 0:
                    dir_x /= length
                    dir_y /= length

                    extension_length = r - np.linalg.norm(np.array(circle_center) - np.array(centroid)) + 0.1 * r

                    dx = centroid[0] - circle_center[0]
                    dy = centroid[1] - circle_center[1]
                    distance2 = np.sqrt(dx*dx + dy*dy)

                    if distance2 > 0:

                        dx_norm = dx / distance2
                        dy_norm = dy / distance2
                        
                        extended_x = int(circle_center[0] + dx_norm * (distance2 + extension_length))
                        extended_y = int(circle_center[1] + dy_norm * (distance2 + extension_length))
                        extended_point = (extended_x, extended_y)
                        
                        global_extended_point = extended_point
                        global_circle_center = circle_center
                        global_black_foreground = black_foreground

                        cv2.line(black_foreground, circle_center, extended_point, (0,255, 255), 1)

                        for i, (reg_circle_center, reg_extended_point) in enumerate(list_positions):
                            cv2.line(black_foreground, reg_circle_center, reg_extended_point, (255, 255, 255), 1)
                            cv2.putText(black_foreground, str(i+1), (reg_extended_point[0] + 10, reg_extended_point[1]), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        
                        yellow_line_intersection_x = int(circle_center[0] + dx_norm * circle_radius)
                        yellow_line_intersection_y = int(circle_center[1] + dy_norm * circle_radius)
                        yellow_line_intersection = (yellow_line_intersection_x, yellow_line_intersection_y)

                        cv2.drawMarker(black_foreground, yellow_line_intersection,(0,0,255),cv2.MARKER_SQUARE, 20, 1, 1)
                        
                        cropped_intersection = cropped_frame[yellow_line_intersection_y-30:yellow_line_intersection_y + 30, 
                                                     yellow_line_intersection_x-30:yellow_line_intersection_x + 30]
                        cropped_intersection = cv2.resize(cropped_intersection,None,fx=4,fy=4,interpolation=cv2.INTER_LANCZOS4)
                        
                        #cv2.imshow('CROPPED_INTERSECTION',cropped_intersection)

                        line_vector_x = extended_point[0] - circle_center[0]
                        line_vector_y = extended_point[1] - circle_center[1]

                        angle_rad = math.atan2(line_vector_y, line_vector_x)
                        current_angle_deg_ref = math.degrees(angle_rad)

                        if current_angle_deg_ref < 0: 
                            current_angle_deg_ref += 360
                        elif current_angle_deg_ref == 0 : 
                            current_angle_deg_ref = 0.000001

                    current_angle = display_current_angle()           
                    
                    if len(angles) == 0:
                        cur_reading = []
                    else:
                        cur_reading = round(value_range*(current_angle/max(angles)),2)

                    text_items = [
                        ("elapsed_time",        f"{frame_counter//30}",                         "s",        30),
                        ("delta_angle",         f"{current_angle_deg_ref:.2f}",                 "deg",      60),
                        ("circ_[x,y]",          f"({circle_data[0]}, {circle_data[1]})",        "pixels",   90),
                        ("ref_angle",           f"{round(current_angle,2)}",                    "deg",      120),
                        ("cur_len(fitline)",    f"{len(fitline)}",                              "",         150),
                        ("len_contours",        f"{len(contours)}",                             "",         180),
                        ("rect_area",           f"{(w*h)}",                                     "pixels",   210),
                        ("len(results_list)",   f"{len(list_positions)}",                       "",         360),
                        ("results_list",        f"{results}",                                   "",         390)
                    ]
                    
                    for i, (label, value, unit, y_pos) in enumerate(text_items):
                        text = f"{label}: {value}"
                        if unit:
                            text += f" [{unit}]"
                        
                        cv2.putText(black_foreground, text, (10, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)

                    if (global_v_inicial is not None and global_v_final is not None):
                        calculate_angle(global_circle_center, global_extended_point)
                        
                    cv2.imshow('OUTPUT', frame)
                    cv2.imshow('SCENE', black_foreground)
                    
    return frame, line_coordinates_list

def history_trackbar(val):
    global history
    history = val
    
def varThreshold_trackbar(val):
    global varThreshold
    varThreshold = val

def thresh2_trackbar(val):
    global thresh_value2
    thresh_value2 = val

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

def thresh_trackbar(val):
    global thresh_value
    thresh_value = val

def minLineLength_trackbar(val):
    global minLineLength
    minLineLength = val

def maxLineGap_trackbar(val):
    global maxLineGap
    maxLineGap = val

def register_position():
    global global_black_foreground, global_circle_center, global_extended_point, list_positions, global_last_registered_line
    if (global_black_foreground is not None and global_circle_center is not None and global_extended_point is not None):
        list_positions.append((global_circle_center, global_extended_point))
        global_last_registered_line = (global_circle_center, global_extended_point)
        calculate_all_angles()
             
def calculate_angle(global_circle_center, global_extended_point):
    dx = global_extended_point[0] - global_circle_center[0]
    dy = global_extended_point[1] - global_circle_center[1]

    angle_rad = math.atan2(dy, dx)
    current_angle_deg_ref = math.degrees(angle_rad)
    if current_angle_deg_ref < 0:
        current_angle_deg_ref += 360
        print(f"{current_angle_deg_ref:.0f}")

    return current_angle_deg_ref

def calculate_smallest_angle_between_lines(line1, line2):
    center1, point1 = line1
    center2, point2 = line2
    
    vec1 = (point1[0] - center1[0], point1[1] - center1[1])
    vec2 = (point2[0] - center2[0], point2[1] - center2[1])
    
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    magnitude1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
    
    if magnitude1 == 0 or magnitude2 == 0:
        return 0
    
    cos_theta = dot_product / (magnitude1 * magnitude2)
    
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    angle_rad = math.acos(cos_theta)
    current_angle_deg_ref = math.degrees(angle_rad)
    
    return current_angle_deg_ref

def calculate_smallest_angle_from_list_positions():
    global list_positions
    
    if len(list_positions) < 2:
        print("Need at least 2 registered positions to calculate angle")
        return None
    
    first_line = list_positions[0]
    last_line = list_positions[-1]
    
    smallest_angle = calculate_smallest_angle_between_lines(first_line, last_line)
    
    print(f"Smallest angle between first and last registered positions: {smallest_angle:.2f} degrees")
    return smallest_angle

def angle_to_unit(cur_angle, range_angle, range_value):
    result_ang = round(range_value * (cur_angle/range_angle), res_calc)
    return result_ang

def calculate_all_angles():
    global list_positions, results
    
    if len(list_positions) < 2:
        print(">>> INSUFFICIENT POSITIONS TO CALCULATE ANGLE BETWEEN FEATURES")
        return []
    
    else:
        angles = []
        results = []
        for i in range(len(list_positions)):
            angle = calculate_smallest_angle_between_lines(list_positions[0], list_positions[i])
            angles.append(angle)

        for a in angles:
            result = angle_to_unit(a, max(angles), range_value)
            results.append(result)
    return results

if __name__ == "__main__":

    select_roi_window()

    cap = cv2.VideoCapture(0)
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, override_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, override_height)

    # Create control windows
    #cv2.namedWindow('Control')
    #cv2.createTrackbar('Canny1', 'Control', minradius, 1500, canny1_trackbar)
    #cv2.createTrackbar('Canny2', 'Control', maxradius, 1500, canny2_trackbar)
    #cv2.createTrackbar('history', 'Control', history, 1500, history_trackbar)
#
    #cv2.namedWindow('LSD')
    #cv2.createTrackbar('MinRadius', 'LSD', minradius, 255, minradius_trackbar)
    #cv2.createTrackbar('MaxRadius', 'LSD', maxradius, 255, maxradius_trackbar)
    #cv2.createTrackbar('thresh_value', 'LSD', thresh_value, 1500, thresh_trackbar)
    #cv2.createTrackbar('thresh_value2', 'LSD', thresh_value2, 1500, thresh2_trackbar)
    #cv2.createTrackbar('minLineLength', 'LSD', minLineLength, 1500, minLineLength_trackbar)
    #cv2.createTrackbar('maxLineGap', 'LSD', maxLineGap, 1500, maxLineGap_trackbar)
    #cv2.createTrackbar('H_varThreshold', 'LSD', varThreshold, 650, varThreshold_trackbar)

    global_black_foreground = None
    global_circle_center = None
    global_extended_point = None
    list_positions = []
    start_position = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_with_lines, _ = read(frame)
        
        if global_black_foreground is not None:
            cv2.imshow("SCENE", global_black_foreground)
        
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('q'):
            break

        elif k == 32:  # ESPAÇO
            register_position()
            
        elif k == ord('c'): 
            calculate_all_angles()

        elif k == ord('v'):
            print(results)
            
        elif k == ord('s'):
            print(">>> RESTARTING ROI SELECTION")
            cap.release()
            cv2.destroyAllWindows()
            select_roi_window()
            cap = cv2.VideoCapture(0)
    
        frame_counter += 1
    
    cap.release()
    cv2.destroyAllWindows()