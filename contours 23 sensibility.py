import cv2
import numpy as np
import math

############# FUNCIONAL PARA MOSTRADORES ANALÓGICOS DE ESCALA COM NÚMEROS NATURAIS
############# CONTÉM ERRO DE LINEARIDADE NÃO DETECTADO

minLineLength = 0
maxLineGap = 0

canny1 = 10
canny2 = 1500

thresh_value = 17
thresh_value2 = 9
param1 = 102
param2 = 3 * param1 
dp = 7
minradius = 167
maxradius = 127
distance = 201

history=150
varThreshold=120

gamma = 1

alpha = 1.5
beta = 0    

min_contour_area = 50

#### GLOBAL VALORES PARA REGISTRAR LINHA
global_circle_center = None
global_extended_point = None
global_black_foreground = None

#### GLOBAL VALORES PARA CALCULAR POSIÇÃO ATUAL
global_start_position_c_value = None
global_final_position_c_value = None
global_last_registered_line = None
#relative_angle = 0

#### GLOBAL VALORES PARA ÂNGULO E CONVERSÃO DE UNIDADE DE MEDIDA
global_v_inicial = None
global_v_final = None
range_value = 35

list_positions = []
start_position = []

frame_counter = 0

backSub = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=True)

def display_current_angle():
    global global_last_registered_line, global_circle_center, global_extended_point, list_positions
    
    current_angle_display = 0
    
    if global_last_registered_line is not None and global_circle_center is not None and global_extended_point is not None:
        if len(list_positions) > 0:
            last_line = list_positions[-1]
            last_center, last_point = last_line
            

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


def detect_lines(frame):

    global global_black_foreground, global_circle_center, global_extended_point
    #frame = cv2.resize(frame, (1280, 720))

    black_foreground = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

    #print(frame.shape[0])

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    
    frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)#####TESTE DE CONTRASTE
    
    #medianblur = cv2.medianBlur(frame,3)
    #frame = medianblur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.Canny(gray, canny1, canny2)
    cv2.imshow('Canny', gray)

    line_coordinates_list = []
    all_points_for_fitting = []
    
    hcircle = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, distance, param1=param1, param2=param2, minRadius=minradius, maxRadius=maxradius)
    
    circle_detected = False
    circle_data = None
    
    if hcircle is not None:
        hcircle = np.round(hcircle[0, :]).astype("int")
        
        for (x, y, r) in hcircle:
            circle_data = (x, y, r * 0.9)
            circle_detected = True
            circle_tolerance = int(circle_data[2])

            fg_mask_full = backSub.apply(frame)

            cv2.circle(frame, (x, y), circle_tolerance, (0, 255, 0), 1)
            cv2.circle(frame, (x, y), 2, (0, 255, 0), 1)
            outer_circle = cv2.circle(black_foreground, (x, y), circle_tolerance, (0, 255, 0), 1)
            cv2.circle(black_foreground, (x, y), 2, (0, 255, 0), 1)
            cv2.circle(mask, (x, y), circle_tolerance, 255, -1)###MÁSCARA PARA ISOLAR ELEMENTOS DENTRO DO CÍRCULO
            
            fg_mask_circular = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)
            retval, mask_thresh = cv2.threshold(fg_mask_circular, 180, 500, cv2.THRESH_BINARY)
            cv2.imshow('mask_thresh',fg_mask_circular)
            mask_thresh = cv2.medianBlur(mask_thresh,3)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1,1))
            mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
            contours, hierarchy = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            roi_mask = np.zeros_like(frame)
            #cv2.circle(roi_mask, (x, y), r, (255, 255, 255), -1)    ####ESSE CÍRCULO ATRAPALHAVA PARA DETERMINAR A DIREÇÃO DA LINHA
            frame_circle_only = cv2.bitwise_and(frame, roi_mask)
            frame_ct = frame.copy()
            cv2.drawContours(frame_ct, contours, -1, (0, 255, 0), 2) #### APLICA MÁSCARA PARA IDENTIFICAR COISAS DENTRO DO CÍRCULO
            frame_ct_circular = cv2.bitwise_and(frame_ct, roi_mask)  ###APLICA MÁSCARA GRÁFICA PRETA DO LADO DE FORA DO CÍRCULO
            
            large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
            rectangle_mask = np.zeros_like(mask_thresh)
            
            for cnt in large_contours: #### ESSA REGIÃO AQUI CUIDA DOS RETÂNGULOS VERMELHOS
                x_rect, y_rect, w, h = cv2.boundingRect(cnt)
                
                contour_center = (x_rect + w//2, y_rect + h//2)
                dist_from_circle_center = np.sqrt((contour_center[0] - x)**2 + (contour_center[1] - y)**2)#### RAIO DO CÍRCULO IDENTIFICADO
                
                if dist_from_circle_center <= r:

                    cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (255, 255, 255), 1)
                    cv2.rectangle(rectangle_mask, (x_rect, y_rect), (x_rect + w, y_rect + h), 255, -1)
                    cv2.imshow('Control', 0)
    
            #mask_thresh = cv2.Canny(mask_thresh, canny1, canny2)
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
        
        [vx_all, vy_all, x0_all, y0_all] = cv2.fitLine(fitline, cv2.DIST_WELSCH, 0, 0.01, 0.01)
        
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
            
            cv2.line(frame, left_all, right_all, (0, 255, 255), 1) ##LINHA CORRETA
            #cv2.line(black_foreground, left_all, right_all, (0, 0,255), 1) ##LINHA CORRETA
            #cv2.line(black_foreground, (circle_data[0],circle_data[1]), right_all, (0,255,0), 1) ##LINHA CORRETA
            #print(left_all,right_all) 

            ##CENTRÓIDE DOS PONTOS QUE COMPÕEM A LINHA AJUSTADA
            centroid = (int(x0_all), int(y0_all))
            cv2.circle(frame, centroid, 5, (0, 255, 0), 1)
            cv2.circle(black_foreground, centroid, 5, (0, 255, 0), 1)
            ###SERIA INTERESSANTE MEDIR SE O CENTRÓIDE ESTÁ OK POR MEIO DE MEDIR A DISTÂNCIA DELE ATÉ O CENTRO DO CÍRCULO, EM RELAÇÃO AO RAIO DO CÍRCULO

            if circle_detected and circle_data and needle_origin and needle_end:
                circle_center = (circle_data[0], circle_data[1])
                circle_radius = circle_data[2]
                
                dir_x = needle_end[0] - needle_origin[0]
                dir_y = needle_end[1] - needle_origin[1]
                
                length = np.sqrt(dir_x*dir_x + dir_y*dir_y)
                if length > 0:
                    dir_x /= length
                    dir_y /= length
                    
                    intersection_x = int(circle_center[0] + dir_x * circle_radius)
                    intersection_y = int(circle_center[1] + dir_y * circle_radius)
                    intersection = (intersection_x, intersection_y)
                    
                    #cv2.line(frame, circle_center, centroid, (0, 255, 255), 1) ###LINHA CONFIAVEL
                    
                    #cv2.drawMarker(frame, intersection,(0,0,255),cv2.MARKER_SQUARE,20,2,1) ###QUADRADO VERMELHO INTERSECÇÃO

                    extension_length = r - np.linalg.norm(np.array(circle_center) - np.array(centroid)) + 0.1 * r ###CALCULA O QUANTO ESTENDER A LINHA, DADO POR r - DIST(CENTRO E CENTRÓIDE DE FITLINE) + 10% RAIO

                    dx = centroid[0] - circle_center[0]
                    dy = centroid[1] - circle_center[1]
                    distance2 = np.sqrt(dx*dx + dy*dy)

                    if distance2 > 0:

                        dx_norm = dx / distance2
                        dy_norm = dy / distance2
                        
                        extended_x = int(circle_center[0] + dx_norm * (distance2 + extension_length))
                        extended_y = int(circle_center[1] + dy_norm * (distance2 + extension_length))
                        extended_point = (extended_x, extended_y)
                        
                        ####DETERMINAR GLOBAIS PARA REGISTRAR LINHA DE REF
                        global_extended_point = extended_point
                        global_circle_center = circle_center
                        global_black_foreground = black_foreground

                        cv2.line(black_foreground, circle_center, extended_point, (0,255, 255), 1)####LINHA AMARELA DO CENTRO DO INDICADOR ATÉ A EXTENSÃO DA LINHA DO VETOR DO CENTRÓIDE ATÉ A BORDA DO CÍRCULO

                        for i, (reg_circle_center, reg_extended_point) in enumerate(list_positions): ####ENUMERA A QUANTIDADE DE REGISTROS E DESENHA LINHA
                            cv2.line(black_foreground, reg_circle_center, reg_extended_point, (255, 255, 255), 1) ####LINHA BRANCA PARA MARCAR REGISTRO
                            #cv2.circle(black_foreground, reg_extended_point, 5, (0, 255, 255), 1) ###CÍRCULO DESNECESSÁRIO MAS VOU DEIXAR ELE AÍ
                            cv2.putText(black_foreground, str(i+1), (reg_extended_point[0] + 10, reg_extended_point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1) ###ÍNDICE DE LINHA
                        
                        yellow_line_intersection_x = int(circle_center[0] + dx_norm * circle_radius)
                        yellow_line_intersection_y = int(circle_center[1] + dy_norm * circle_radius)
                        yellow_line_intersection = (yellow_line_intersection_x, yellow_line_intersection_y)

                        cv2.drawMarker(black_foreground, yellow_line_intersection,(0,0,255),cv2.MARKER_SQUARE,20,1,1) ####CÍRCULO DA INTERSECCÇÃO EM BLACK_FOREGROUND

                    line_vector_x = extended_point[0] - circle_center[0]
                    line_vector_y = extended_point[1] - circle_center[1]

                    angle_rad = math.atan2(line_vector_y, line_vector_x)
                    current_angle_deg_ref = math.degrees(angle_rad)

                    if current_angle_deg_ref < 0: current_angle_deg_ref += 360
                    if current_angle_deg_ref == 0 : current_angle_deg_ref = 0.000001

                    if (global_extended_point is not None and global_circle_center is not None and global_start_position_c_value is not None):

                        relative_angle = calculate_smallest_angle_between_lines((global_extended_point,global_circle_center),global_start_position_c_value)
                        print('relative_angle:', relative_angle)
                    else:
                        relative_angle = 0

                    current_angle = display_current_angle()
                    teste_relative_value = current_angle/current_angle_deg_ref

                    text_items = [
                        ("elapsed_time",        f"{frame_counter//30}",                     "s",        30),
                        #("frame_counter",      f"{frame_counter}",                         "frames",   50),
                        ("len(list_pos)",       f"{len(list_positions)}",                   "",         50),
                        ("cur_angle",           f"{current_angle_deg_ref:.2f}",             "deg",      110),
                        ("circ_data",           f"({circle_data[0]}, {circle_data[1]})",    "pixels",   130),
                        ("rel_angle",           f"{round(current_angle,2)}",                "deg",      150),
                        ("cur_reading",         f"{round(35*(current_angle/110),2)}",       "mm",       170),
                        ("cur_len(fitline)",    f"{len(fitline)}",                          "",         190)
                    ]

                    for i, (label, value, unit, y_pos) in enumerate(text_items):
                        text = f"{label}: {value}"
                        if unit:
                            text += f" [{unit}]"
                        
                        cv2.putText(black_foreground, text, (10, y_pos), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                    
                    if (global_v_inicial is not None and global_v_final is not None):
                        calculate_angle(global_circle_center, global_extended_point)
                        
                    cv2.imshow('PRE_INTERSECTION', frame)
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

####CONTROLES PARA O LSD
def thresh_trackbar(val):
    global thresh_value
    thresh_value = val

def minLineLength_trackbar(val):
    global minLineLength
    minLineLength = val

def maxLineGap_trackbar(val):
    global maxLineGap
    maxLineGap = val

cap = cv2.VideoCapture(0)

cv2.namedWindow('Control')
cv2.createTrackbar('MinRadius', 'Control', minradius, 255, minradius_trackbar)
cv2.createTrackbar('MaxRadius', 'Control', maxradius, 255, maxradius_trackbar)
cv2.createTrackbar('Canny1', 'Control', minradius, 1500, canny1_trackbar)
cv2.createTrackbar('Canny2', 'Control', maxradius, 1500, canny2_trackbar)

cv2.createTrackbar('varThreshold', 'Control', varThreshold, 100, varThreshold_trackbar)
cv2.createTrackbar('history', 'Control', history, 1500, history_trackbar)

cv2.namedWindow('LSD')
cv2.createTrackbar('thresh_value', 'LSD', thresh_value, 1500, thresh_trackbar)
cv2.createTrackbar('minLineLength', 'LSD', minLineLength, 1500, minLineLength_trackbar)
cv2.createTrackbar('maxLineGap', 'LSD', maxLineGap, 1500, maxLineGap_trackbar)

#cv2.namedWindow('Canny')
#cv2.createTrackbar('Canny1', 'Canny', canny1, 1000, canny1_trackbar)
#cv2.createTrackbar('Canny2', 'Canny', canny2, 1000, canny2_trackbar)

global_black_foreground = None
global_circle_center = None
global_extended_point = None
list_positions = []
start_position = []

def register_position():
    global global_black_foreground, global_circle_center, global_extended_point, list_positions, global_last_registered_line
    if (global_black_foreground is not None and global_circle_center is not None and global_extended_point is not None):
        list_positions.append((global_circle_center, global_extended_point))
        global_last_registered_line = (global_circle_center, global_extended_point)
            
def calculate_angle(global_circle_center, global_extended_point):
    dx = global_extended_point[0] - global_circle_center[0]
    dy = global_extended_point[1] - global_circle_center[1]

    angle_rad = math.atan2(dy, dx)
    current_angle_deg_ref = math.degrees(angle_rad)
    if current_angle_deg_ref < 0:
        current_angle_deg_ref += 360
        print(f"{current_angle_deg_ref:.0f}")

    return current_angle_deg_ref

def register_start_position():
    global global_black_foreground, global_circle_center, global_extended_point, list_positions
    start_position_c_value = list_positions[0]
    #if (global_black_foreground is not None and global_circle_center is not None and global_extended_point is not None):
    #    start_position.append((global_circle_center, global_extended_point))
    #    start_position_c_value = start_position[-1]
        
    print(start_position)
    #v_inicial = input("Valor inicial: ")

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

    result = range_value * (cur_angle/range_angle)
    return result

def calculate_all_angles():
    global list_positions
    
    if len(list_positions) < 2:
        print(">>> INSUFICIENT POSITIONS TO CALCULATE ANGLE BETWEEN FEATURES")
        return []
    
    angles = []
    for i in range(len(list_positions) - 1):
        angle = calculate_smallest_angle_between_lines(list_positions[i], list_positions[i + 1])
        angles.append(angle)
        #angle_to_unit(angle)

        print(f"Ângulo {i} e {i+1}: {angle:.0f} °")

        print(f"Resultado {i} e {i+1}: {angle_to_unit(angle, max(angles), range_value)} mm")#####USA A FUNÇÃO PARA CONVERTER O ÂNGULO ENTRE LINHAS E O RANGE DECLARADO COMO CONSTANTE PARA CALCULAR O VALOR DE MEDIÇÃO
    
    return angles

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_with_lines, _ = detect_lines(frame)
    
    #cv2.imshow("Final", frame_with_lines)
    if global_black_foreground is not None:
        cv2.imshow("SCENE", global_black_foreground)
    
    k = cv2.waitKey(1) & 0xFF
    
    if k == ord('q'):
        break
    elif k == 32:  # ESPAÇO
        register_position()
        print(f"Position registered. Total: {len(list_positions)}")
    
    elif k == ord('a'):  # A
        register_start_position()
    
    elif k == ord('s'):  # S
        if len(list_positions) >= 2:
            smallest_angle = calculate_smallest_angle_from_list_positions()
    
    elif k == ord('c'):  # C 
        calculate_all_angles()
    
    elif k == ord('p'):  # P
        print(f"Registered positions: {len(list_positions)}")
        for i, pos in enumerate(list_positions):
            center, point = pos
            angle = calculate_angle(center, point)
            print(f"Position {i}: Center={center}, Point={point}, Angle={angle:.2f}°")
    
    #frame_counter +=1
    

cap.release()
cv2.destroyAllWindows()