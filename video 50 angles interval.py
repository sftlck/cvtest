import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import matplotlib.pyplot as plt

# =======================
# CLASSES DE CONFIGURAÇÃO
# ============================================================================

@dataclass
class CameraParams:
    """Camera calibration parameters"""
    fx:                 float = 426.7231198260606
    fy:                 float = 424.44185129184774
    skew:               float = 0.0
    cx:                 float = 308.3359965293331
    cy:                 float = 233.71108395163847
    width:              int =   640
    height:             int =   480
    r0:                 int =   2316661
    r1:                 int =   -32512066

@dataclass
class ProcessingParams:
    """Image processing parameters"""
    n_regions:          int =   40
    min_region_area:    int =   10

    canny1:             int =   10
    canny2:             int =   255

    thresh_value:       int =   214
    thresh_value2:      int =   255

    dp:                 int =   7
    param1:             int =   102
    param2:             int =   306 
    min_radius:         int =   90
    max_radius:         int =   100
    distance:           int =   201

    history:            int =   300
    var_threshold:      int =   120

    gamma:              float = 1.0
    alpha:              float = 1.5
    beta:               float = 0.0
    
    min_contour_area:   int =   15
    
    ring_scale_factor:  float = 0.8
    ring_thresh_low:    int =   170
    ring_thresh_high:   int =   255
    
    value_range:        float = 35.0
    resolution:         float = 0.1
    
    num_bins:           int =   360
    threshold_value:    int =   127
    limit_chart:        int =   25

@dataclass
class ROIConfig:
    """Region of Interest configuration"""
    center_x:           int = 326
    center_y:           int = 267
    radius:             int = 118
    selected:           bool = True

@dataclass
class CalibrationData:
    """Store MIN/MAX calibration lines"""
    min_line:           Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # (centroid, center)
    max_line:           Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    min_value:          float = 0.0
    max_value:          float = 100.0
    is_calibrated:      bool = False
    frozen_frame:       Optional[np.ndarray] = None

# ========================
# GEARENCIMENTO DE ESTADOS
# ============================================================================

class ProcessingState:
    """Global processing state"""
    
    def __init__(self):
        self.circle_center:         Optional[Tuple[int, int]] = None
        self.extended_point:        Optional[Tuple[int, int]] = None
        self.black_foreground:      Optional[np.ndarray] = None
        
        self.registered_positions:  List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        self.last_registered_line:  Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        
        self.angles:                List[float] = []
        self.results:               List[float] = []
        
        self.frame_counter:         int = 0
        
        self.calibration =          CalibrationData()
        self.calibration_mode =     False
        self.selection_stage =      0  

# =========================================
# CLASSE PARA CORRIGIR DISTORÇÕES NA CÂMERA
# ============================================================================

class CameraCalibrator:
    """Handles camera calibration and undistortion"""
    
    def __init__(self, params: CameraParams):
        self.params = params
        self.camera_matrix = self._create_camera_matrix()
        self.dist_coeffs = self._create_distortion_coefficients()
        self.map1, self.map2, self.roi, self.new_camera_matrix = self._setup_undistort_maps()
    
    def _create_camera_matrix(self) -> np.ndarray:
        """Create camera matrix from parameters"""
        return np.array([
            [self.params.fx, self.params.skew, self.params.cx],
            [0, self.params.fy, self.params.cy],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _create_distortion_coefficients(self) -> np.ndarray:
        """Create distortion coefficients from parameters"""
        scale_factor = 1e9
        k1 = self.params.r0 / scale_factor
        k2 = self.params.r1 / scale_factor
        return np.array([k1, k2, 0, 0, 0], dtype=np.float32)
    
    def _setup_undistort_maps(self) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int, int, int], np.ndarray]:
        """Setup undistortion maps"""
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(self.camera_matrix, self.dist_coeffs, (self.params.width, self.params.height), 1, (self.params.width, self.params.height))
        map1, map2 = cv2.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, new_camera_matrix, (self.params.width, self.params.height), cv2.CV_16SC2)
        return map1, map2, roi, new_camera_matrix
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion to frame"""
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LANCZOS4)
        x, y, w, h = self.roi
        return undistorted[y:y+h, x:x+w]

# =========================+
# IMAGE PROCESSING UTILITIES
# ============================================================================

class ImageProcessor:
    """Core image processing utilities"""
    
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.processing_params = ProcessingParams()
        self.app = app
        self.last_component_centroids = []
        self.last_offset = None
        self.angle_calculator = AngleCalculator(self.processing_params.value_range, self.processing_params.resolution)
    
    def create_polar_histogram(self, frame: np.ndarray, ax, gauge_center: Optional[Tuple[int, int]] = None, scale_factor: float = 1.0, target_img: Optional[np.ndarray] = None, result_frame: Optional[np.ndarray] = None, offset: Optional[Tuple[int, int]] = None):
        
        img_copy = frame.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.params.threshold_value, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
    
        black_regions = []
        component_centroid = []
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]

            if area >= 5:
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                
                black_regions.append({                                                                  ###### AQUI AINDA TEM O PROBLEMA DO PONTEIRO TAMBÉM SER CONSIDERADO UM ELEMENTO VISÍVEL, TEM QUE TER UM BUGFIX
                    'label': label, 'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'black_pixels': area,
                    'centroid': (int(centroids[label][0]), int(centroids[label][1]))
                })
                component_centroid.append((int(centroids[label][0]), int(centroids[label][1])))
        
        self.last_component_centroids = component_centroid.copy()
        self.last_offset = offset
        
        black_regions.sort(key=lambda x: x['black_pixels'], reverse=True)
        connected_results = black_regions
        
        if len(connected_results) > 0:
            display_img = img_copy.copy()
            for i, region in enumerate(connected_results[:25]):
                cv2.rectangle(display_img,
                             (region['x'], region['y']),
                             (region['x'] + region['width'], region['y'] + region['height']),
                             (255, 0, 0), 1)
            cv2.imshow('detected_regions', display_img)

        if self.app and hasattr(self.app, 'current_result_frame'):

            result_frame = self.app.current_result_frame
            frame = self.app.frame
            if result_frame is not None and offset is not None and hasattr(self.app, 'circle_data'):
                circle_data = self.app.circle_data
                center_x, center_y, radius = circle_data[0], circle_data[1], circle_data[2]
                offset_x, offset_y = offset
                
                for i, (x, y) in enumerate(component_centroid):
                    original_x = offset_x + x
                    original_y = offset_y + y
                    
                    cv2.line(result_frame,      (original_x, original_y), (center_x, center_y), (125, 125, 125), 1)
                    cv2.line(frame,             (original_x, original_y), (center_x, center_y), (0, 0, 125), 1)
                    #cv2.putText(result_frame,   str(i), (int(original_x), int(original_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1) #####ÍNDICE DE LINHA
                    
                    dx = center_x - original_x
                    dy = center_y - original_y

                    dist = math.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        
                        extended_x = int(center_x + dx_norm * (radius + 0.2 * radius))
                        extended_y = int(center_y + dy_norm * (radius + 0.2 * radius))

                        intersections = self.angle_calculator.find_line_circle_intersections((original_x, original_y),
                                                                                             (extended_x, extended_y),
                                                                                             (center_x, center_y),
                                                                                             radius)
                        
                        #cv2.line(result_frame, (center_x, center_y), (extended_x,extended_y), (200, 200, 200))
                                           
                        #cv2.line(result_frame, (center_x, center_y), (extended_x, extended_y), (255, 255, 0), 1)
                        
                        if intersections:
                            intersections_with_dist = []
                            for ix, iy in intersections:
                                dist_from_centroid = math.sqrt((ix - original_x)**2 + (iy - original_y)**2)
                                intersections_with_dist.append((dist_from_centroid, (ix, iy)))
                            
                            intersections_with_dist.sort(reverse=True)
                            if intersections_with_dist:
                                far_intersection = intersections_with_dist[0][1]
                                cv2.circle(result_frame, far_intersection, 6, (0, 255, 255), 1)                         ##### CÍRCULO QUE MOSTRA A INTERSECÇÃO DA MARCA DA ESCALA E A BORDA
                                
                                #cv2.putText(result_frame, str(i), (far_intersection[0] + 10, far_intersection[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                        if self.app and hasattr(self.app, 'min_line'):                                                                  ########### AQUI A GENTE HERDA X0,Y0 E X1,Y1 DA LINHA MÍNIMA E MÁXIMA
                            min_line = self.app.min_line
                            #print(min_line)

                            min_lineA = self.angle_calculator.min_angle_between_lines(((center_x, center_y), (original_x,original_y)),  min_line)

                        if self.app and hasattr(self.app, 'max_line'):                                                                  ########### AQUI A GENTE HERDA X0,Y0 E X1,Y1 DA LINHA MÍNIMA E MÁXIMA
                            max_line = self.app.max_line
                            #print(max_line)

                            max_lineA = self.angle_calculator.angle_between_lines(((center_x, center_y), (original_x,original_y)),  min_line)

                            #if i == 17:                                                                                                        #### SÓ PARA DEBUG
                            #    cv2.putText(result_frame, str(int(res)), far_intersection, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                        
                            min_scale = self.angle_calculator.min_angle_between_lines(max_line,min_line)
                            max_scale = self.angle_calculator.max_angle_between_lines(max_line,min_line)
                            actual_scale = self.angle_calculator.min_angle_between_lines(min_line,((center_x, center_y), (original_x,original_y)))

                            print(min_scale, min_lineA, min_scale)
                            if min_scale <= actual_scale and actual_scale <= max_scale:
                                cv2.putText(result_frame, str(int(actual_scale)), (original_x,original_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
                                
        return black_mask

    def find_nearest_region(self, click_x: int, click_y: int, component_centroids: List[Tuple[int, int]], offset: Tuple[int, int], max_distance: int = 50) -> Optional[Tuple[int, int]]:
        if not component_centroids or offset is None:
            return None
        
        crop_click_x = click_x - offset[0]
        crop_click_y = click_y - offset[1]
        
        min_dist = float('inf')
        closest_centroid = None
        
        for cx, cy in component_centroids:
            dist = np.sqrt((cx - crop_click_x)**2 + (cy - crop_click_y)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest_centroid = (cx, cy)
        
        return closest_centroid

    @staticmethod
    def find_black_regions(binary_mask: np.ndarray, min_area: int) -> List[Dict]:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
        
        regions = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            if area >= min_area:
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                
                regions.append({
                    'label': label, 'x': x, 'y': y, 'width': w, 'height': h,
                    'area': area, 'centroid': (int(centroids[label][0]), int(centroids[label][1]))
                })
        
        regions.sort(key=lambda x: x['area'], reverse=True)
        return regions
    
    @staticmethod
    def crop_ring_area(image: np.ndarray, center_x: int, center_y: int,
                       inner_r: int, outer_r: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Crop ring area between inner and outer radius and return offset"""
        hh, ww = image.shape[:2]
        white_background = np.ones((hh, ww, 3), dtype=np.uint8) * 255
        
        mask_ring = np.zeros((hh, ww), dtype=np.uint8)
        cv2.circle(mask_ring, (center_x, center_y), outer_r, 255, -1)
        cv2.circle(mask_ring, (center_x, center_y), inner_r, 0, -1)
        
        ring_from_original = cv2.bitwise_and(image, image, mask=mask_ring)
        
        mask_bg = cv2.bitwise_not(mask_ring)
        white_bg = cv2.bitwise_and(white_background, white_background, mask=mask_bg)
        final_image = cv2.add(ring_from_original, white_bg)

        x1 = max(0, center_x - outer_r)
        y1 = max(0, center_y - outer_r)
        x2 = min(ww, center_x + outer_r)
        y2 = min(hh, center_y + outer_r)
        
        cropped = final_image[y1:y2, x1:x2]
        offset = (x1, y1)
        
        return cropped, offset
    
    @staticmethod
    def enhance_image(frame: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """Apply contrast and brightness enhancement"""
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    @staticmethod
    def convert_to_gray(frame: np.ndarray) -> np.ndarray:
        """Convert frame to grayscale"""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

# ===========
# RING PROCESSOR
# ============================================================================

class RingProcessor:
    def __init__(self, params: ProcessingParams, needle_detector: 'NeedleDetector'):
        self.params = params
        self.needle_detector = needle_detector
        self.app = needle_detector.app
        self.image_processor = ImageProcessor(params, app=self.app)
        
        self.fig = None
        self.ax = None
        plt.ion()
        self.fig, self.ax = plt.subplots(1, 1, figsize=(5, 5), subplot_kw={'projection': 'polar'})

    def process(self, ring_img: np.ndarray, n_regions: int, result_frame=None, offset=None):
        gray = self.image_processor.convert_to_gray(ring_img)
        _, output_thresh = cv2.threshold(gray, self.params.ring_thresh_low,
                                        self.params.ring_thresh_high, cv2.THRESH_BINARY)
        
        self.image_processor.create_polar_histogram(ring_img, self.ax, target_img=ring_img,
                                                    result_frame=result_frame, offset=offset)
        return output_thresh

# ======
# NEEDLE DETECTOR
# ============================================================================

class NeedleDetector:
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.app = app
        self.ring_processor = RingProcessor(params, self)
        self.backSub = cv2.createBackgroundSubtractorMOG2(history=params.history,
                                                          varThreshold=params.var_threshold,
                                                          detectShadows=True)
    
    def detect_circle(self, gray: np.ndarray, roi_config: Optional[ROIConfig] = None) -> Optional[Tuple[int, int, int]]:
        """Detect circular gauge using Hough transform or ROI"""
        if roi_config and roi_config.selected:
            return (roi_config.center_x, roi_config.center_y, roi_config.radius)
        
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, self.params.dp, self.params.distance, param1=self.params.param1, param2=self.params.param2, minRadius=self.params.min_radius, maxRadius=self.params.max_radius)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            return (x, y, int(r * 0.9))
        
        return None
    
    def extract_needle_points(self, frame: np.ndarray, circle_data: Tuple[int, int, int],
                             mask: np.ndarray) -> List[List[int]]:
        """Extract points belonging to the needle"""
        x, y, r = circle_data
        
        fg_mask_full = self.backSub.apply(frame)
        
        cv2.circle(mask, (x, y), r, 255, -1)
        fg_mask_circular = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)
        
        _, mask_thresh = cv2.threshold(fg_mask_circular, self.params.thresh_value,
                                       self.params.thresh_value2, cv2.THRESH_BINARY)
        
        cv2.imshow('mask_thresh', mask_thresh)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.params.min_contour_area]
        
        rectangle_mask = np.zeros_like(mask_thresh)
        
        all_points = []
        for cnt in large_contours:
            x_rect, y_rect, w, h = cv2.boundingRect(cnt)
            contour_center = (x_rect + w//2, y_rect + h//2)
            dist = np.sqrt((contour_center[0] - x)**2 + (contour_center[1] - y)**2)
            
            if dist <= r:
                cv2.rectangle(frame, (x_rect, y_rect), (x_rect + w, y_rect + h), (0, 0, 255), 1)
                cv2.rectangle(rectangle_mask, (x_rect, y_rect), (x_rect + w, y_rect + h), 255, -1)
        
        masked_canny = cv2.bitwise_and(mask_thresh, rectangle_mask)
        indices = np.where(masked_canny > 0)
        
        for px, py in zip(indices[1], indices[0]):
            dist = np.sqrt((px - x)**2 + (py - y)**2)
            if py < rectangle_mask.shape[0] and px < rectangle_mask.shape[1] and \
               rectangle_mask[py, px] > 0 and dist <= r:
                all_points.append([px, py])
        
        return all_points
    
    def extend_line(self, point: Tuple[int, int], circle_center: Tuple[int, int], circle_radius: int):
        dx = point[0] - circle_center[0]
        dy = point[1] - circle_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        if distance <= 0:
            return None
        dx_norm = dx / distance
        dy_norm = dy / distance
        extension = circle_radius - distance + circle_radius
        extended_x = int(circle_center[0] + dx_norm * (distance + extension))
        extended_y = int(circle_center[1] + dy_norm * (distance + extension))
        return (point, (extended_x, extended_y))

    def fit_line_to_points(self, points: List[List[int]], circle_center: Tuple[int, int],
                          circle_radius: int) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Fit a line to detected points and extend to needle tip"""
        if len(points) < 20:
            return None
        
        fitline = np.array(points, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(fitline, cv2.DIST_L2, 0, 0.01, 0.01)
        
        if abs(vx[0]) <= 1e-6:
            return None
        
        centroid = (int(x0[0]), int(y0[0]))
        
        dx = centroid[0] - circle_center[0]
        dy = centroid[1] - circle_center[1]
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance <= 0:
            return None
        
        dx_norm = dx / distance
        dy_norm = dy / distance
        
        extension = circle_radius - distance + 0.1 * circle_radius
        extended_x = int(circle_center[0] + dx_norm * (distance + extension))
        extended_y = int(circle_center[1] + dy_norm * (distance + extension))
        
        return (centroid, (extended_x, extended_y))
    
    def process_ring(self, frame, circle_data, n_regions, result_frame=None):
        x, y, r = circle_data
        inner_radius = int(r * self.params.ring_scale_factor)
        if inner_radius < r:
            ring_cropped, offset = self.ring_processor.image_processor.crop_ring_area(
                frame, x, y, inner_radius, r
            )
            if ring_cropped is not None and ring_cropped.size > 0:
                self.ring_processor.process(ring_cropped, n_regions,
                                           result_frame=result_frame, offset=offset)

# ================
# ANGLE CALCULATOR
# ============================================================================

class AngleCalculator:
    
    def __init__(self, value_range: float, resolution: float):
        self.value_range = value_range
        self.resolution = resolution
        self.res_calc = self._calculate_resolution_precision(resolution)
    
    @staticmethod
    def _calculate_resolution_precision(resolution: float) -> int:
        res_str = str(resolution)
        return res_str.find("1") - 1
    
    @staticmethod
    def calculate_angle_from_vector(dx: float, dy: float) -> float:
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        elif angle_deg == 0:
            angle_deg = 0.000001
        return angle_deg
    
    def calculate_needle_angle(self, center: Tuple[int, int], tip: Tuple[int, int]) -> float:
        dx = tip[0] - center[0]
        dy = tip[1] - center[1]
        return self.calculate_angle_from_vector(dx, dy)
    
    @staticmethod
    def angle_between_lines(line1: Tuple[Tuple[int, int], Tuple[int, int]], line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        center1, point1 = line1
        center2, point2 = line2
        
        vec1 = (point1[0] - center1[0], point1[1] - center1[1])
        vec2 = (point2[0] - center2[0], point2[1] - center2[1])
        
        dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0
        
        cos_theta = min(-1.0, min(1.0, dot / (mag1 * mag2)))
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def min_angle_between_lines(line1: Tuple[Tuple[int, int], Tuple[int, int]], 
                                line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """Returns the smallest angle between two lines (0-180°)"""
        center1, point1 = line1
        center2, point2 = line2
        
        vec1 = (point1[0] - center1[0], point1[1] - center1[1])
        vec2 = (point2[0] - center2[0], point2[1] - center2[1])
        
        dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = np.sqrt(vec1[0]**2 + vec1[1]**2)
        mag2 = np.sqrt(vec2[0]**2 + vec2[1]**2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        cos_theta = dot / (mag1 * mag2)
        cos_theta = max(-1.0, min(1.0, cos_theta))
        
        return math.degrees(math.acos(cos_theta))

    @staticmethod
    def max_angle_between_lines(line1: Tuple[Tuple[int, int], Tuple[int, int]], 
                                line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """Returns the larger angle between two lines (180-360°)"""
        acute = AngleCalculator.min_angle_between_lines(line1, line2)
        return 360 - acute

    def angle_to_value(self, angle: float, max_angle: float) -> float:
        if max_angle == 0:
            return 0
        return round(self.value_range * (angle / max_angle), self.res_calc)
    
    @staticmethod  
    def find_line_circle_intersections(line_start: Tuple[int, int],  
                                       line_end: Tuple[int , int],  
                                       circle_center: Tuple[int, int],  
                                       circle_radius: int) -> List[Tuple[int, int]]:
    
        x1, y1 = line_start
        x2, y2 = line_end
        cx, cy = circle_center
        r = circle_radius
        
        dx = x2 - x1
        dy = y2 - y1
        
        fx = x1 - cx
        fy = y1 - cy
        
        a = dx*dx + dy*dy
        b = 2*(fx*dx + fy*dy)
        c = (fx*fx + fy*fy) - r*r
        
        discriminant = b*b - 4*a*c
        
        intersections = []
        
        if discriminant >= 0 and a != 0:
            sqrt_disc = math.sqrt(discriminant)
            
            t1 = (-b - sqrt_disc) / (2*a)
            t2 = (-b + sqrt_disc) / (2*a)
            
            for t in [t1, t2]:
                if 0 <= t <= 1:
                    ix = int(x1 + t * dx)
                    iy = int(y1 + t * dy)
                    intersections.append((ix, iy))
        
        return intersections
# ===========
# VISUALIZATION
# ============================================================================

class Visualizer:
    
    @staticmethod
    def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int] = (0, 255, 0)):
        cv2.circle(frame, center, radius, color, 1)
        cv2.circle(frame, center, 2, color, 1)
    
    @staticmethod
    def draw_needle(frame: np.ndarray, center: Tuple[int, int], tip: Tuple[int, int], color: Tuple[int, int, int] = (0, 255, 255)):
        cv2.line(frame, center, tip, color, 1)
    
    @staticmethod
    def draw_intersection_marker(frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255)):
        cv2.drawMarker(frame, point, color, cv2.MARKER_SQUARE, 20, 1, 1)
    
    @staticmethod
    def draw_registered_positions(frame: np.ndarray, positions: List[Tuple[Tuple[int, int], Tuple[int, int]]]):
        for i, (center, tip) in enumerate(positions):
            cv2.line(frame, center, tip, (255, 255, 255), 1)
            cv2.putText(frame, str(i+1), (tip[0] + 10, tip[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    @staticmethod
    def draw_calibration_lines(frame: np.ndarray, calibration: CalibrationData):
        if calibration.min_line:
            centroid, center = calibration.min_line
            cv2.line(frame, centroid, center, (0, 255, 0), 1)                                                             ######### LINHA E CÍRCULO VERDES PARA INÍCIO DA ESCALA
            #cv2.putText(frame, "min", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            cv2.circle(frame, centroid, 5, (0, 255, 0), -1)
        
        if calibration.max_line:
            centroid, center = calibration.max_line
            cv2.line(frame, centroid, center, (0, 0, 255), 1)
            #cv2.putText(frame, "max", (centroid[0] + 10, centroid[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
        
        #if calibration.is_calibrated:
        #    cv2.putText(frame, f"MIN: {calibration.min_value}", (10, 210),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        #    cv2.putText(frame, f"MAX: {calibration.max_value}", (10, 240),
        #               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    
    @staticmethod
    def draw_info_overlay(frame: np.ndarray, info_items: List[Tuple[str, Any, str, int]]):
        for label, value, unit, y_pos in info_items:
            text = f"{label}: {value}"
            if unit:
                text += f" [{unit}]"
            
            cv2.putText(frame, text, (10, y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    
    @staticmethod
    def draw_progress(frame: np.ndarray, current: int, total: int):
        progress = f"Frame: {current}/{total} ({(current/total*100):.1f}%)"
        cv2.putText(frame, progress, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ==============
# MAIN APPLICATION
# ============================================================================

class GaugeReaderApp:
    def __init__(self, video_path: str):
        self.video_path =           video_path
        self.camera_params =        CameraParams()
        self.processing_params =    ProcessingParams()
        self.roi_config =           ROIConfig()
        self.state =                ProcessingState()
        self.current_result_frame = None
        self.frame =                None
        self.circle_data =          None

        self.calibrator =           CameraCalibrator(self.camera_params)
        self.image_processor =      ImageProcessor(self.processing_params, app=self)
        self.needle_detector =      NeedleDetector(self.processing_params, app=self)
        self.angle_calculator =     AngleCalculator(self.processing_params.value_range, self.processing_params.resolution)
        self.visualizer =           Visualizer()
        
        self.cap =                  None
        self.total_frames =         0
        self.fps =                  0
        self.paused =               False

    def process_frozen_frame_for_calibration(self):
        if self.state.calibration.frozen_frame is None:
            print("No frozen frame available")
            return False
        
        gray = self.image_processor.convert_to_gray(self.state.calibration.frozen_frame)
        circle_data = self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            print("Could not detect circle in frozen frame")
            return False
        
        x, y, r = circle_data
        self.circle_data = circle_data
        
        temp_result = np.zeros_like(self.state.calibration.frozen_frame)
        
        inner_radius = int(r * self.processing_params.ring_scale_factor)
        if inner_radius < r:
            ring_cropped, offset = self.image_processor.crop_ring_area(self.state.calibration.frozen_frame, x, y, inner_radius, r)
            
            if ring_cropped is not None and ring_cropped.size > 0:
                self.image_processor.create_polar_histogram( ring_cropped, None, target_img=ring_cropped, result_frame=temp_result, offset=offset)
                
                centroid_count = len(self.image_processor.last_component_centroids)
                #print(f"Found {centroid_count} regions in frozen frame")               ###### ANOTAÇÃO SOBRE QUANTIDADE DE REGIÕES ENCONTRADAS
                return centroid_count > 0
        
        return False

    def enter_calibration_mode(self):
        if self.frame is not None:
            self.state.calibration.frozen_frame = self.frame.copy()
            self.state.calibration_mode = True
            self.state.selection_stage = 1
            self.paused = True
            
            success = self.process_frozen_frame_for_calibration()
            
    def exit_calibration_mode(self):
        self.state.calibration_mode = False
        self.state.selection_stage = 0
        self.paused = False
        print("Calibration cancelled")

    def handle_calibration_click(self, x: int, y: int):
        if not self.state.calibration_mode:
            return
        
        if not hasattr(self.image_processor, 'last_component_centroids'):
            print("No region data available. Processing frame again...")
            self.process_frozen_frame_for_calibration()
            return
        
        component_centroids = self.image_processor.last_component_centroids
        offset = self.image_processor.last_offset
        
        if not component_centroids or offset is None:
            print("No regions detected in this frame")
            return
        
        nearest = self.image_processor.find_nearest_region(x, y, component_centroids, offset)
        
        if nearest is None:
            print("No region found near click. Try clicking closer to a gray line.")
            return
        
        original_x = offset[0] + nearest[0]
        original_y = offset[1] + nearest[1]

        if self.state.selection_stage == 1:                                                                     ####### PRIMEIRO ESTÁGIO ONDE SELECIONAMOS A LINHA DE MÍNIMO
            self.state.calibration.min_line = ((original_x, original_y), self.state.circle_center)
            
            self.min_line = ((original_x, original_y), self.state.circle_center)
            self.state.selection_stage = 2
            
        elif self.state.selection_stage == 2:                                                                     ####### ESTÁGIO ONDE SELECIONAMOS A LINHA DE MÁXIMO
            self.state.calibration.max_line = ((original_x, original_y), self.state.circle_center)
            
            self.max_line = ((original_x, original_y), self.state.circle_center)
            self.state.calibration.is_calibrated = True
            self.state.calibration_mode = False
            self.state.selection_stage = 0
            self.paused = False
            
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        enhanced_frame = self.image_processor.enhance_image(frame, self.processing_params.alpha,
                                                        self.processing_params.beta)
        
        gray = self.image_processor.convert_to_gray(enhanced_frame)
        
        black_foreground = np.zeros_like(frame)
        self.current_result_frame = black_foreground
        self.frame = frame
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        circle_data = self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            return None
        
        x, y, r = circle_data
        self.circle_data = circle_data
        
        self.visualizer.draw_circle(frame, (x, y), r)
        self.visualizer.draw_circle(black_foreground, (x, y), r)
        
        n_regions = cv2.getTrackbarPos('N regions', 'output_thresh')
        
        self.needle_detector.process_ring(enhanced_frame, circle_data, n_regions, result_frame=black_foreground)
        
        if not self.state.calibration_mode:
            points = self.needle_detector.extract_needle_points(enhanced_frame, circle_data, mask)
            
            if points:
                result = self.needle_detector.fit_line_to_points(points, (x, y), r)
                
                if result:
                    centroid, extended_point = result
                    
                    height, width = frame.shape[:2]
                    fitline = np.array(points, dtype=np.float32)
                    [vx, vy, x0, y0] = cv2.fitLine(fitline, cv2.DIST_L2, 0, 0.01, 0.01)
                    
                    if abs(vx[0]) > 1e-6:
                        slope = vy[0] / vx[0]
                        left_y = int(y0[0] + slope * (0 - x0[0]))
                        right_y = int(y0[0] + slope * (width - 1 - x0[0]))
                        cv2.line(frame, (0, left_y), (width - 1, right_y), (0, 255, 255), 1)
                    
                    cv2.circle(frame, centroid, 5, (0, 255, 0), 1)
                    cv2.circle(black_foreground, centroid, 5, (0, 255, 0), 1)
                    
                    self.state.circle_center = (x, y)
                    self.state.extended_point = extended_point
                    self.state.black_foreground = black_foreground.copy()
                    
                    self.visualizer.draw_needle(black_foreground, (x, y), extended_point)
                    self.visualizer.draw_registered_positions(black_foreground,
                                                            self.state.registered_positions)
                    
                    dx = extended_point[0] - x
                    dy = extended_point[1] - y
                    dist = np.sqrt(dx*dx + dy*dy)
                    
                    if dist > 0:
                        dx_norm = dx / dist
                        dy_norm = dy / dist
                        intersection_x = int(x + dx_norm * r)
                        intersection_y = int(y + dy_norm * r)
                        
                        self.visualizer.draw_intersection_marker(black_foreground, (intersection_x, intersection_y))
                        
                        current_angle = self.angle_calculator.calculate_needle_angle((x, y), extended_point)
                        
                        info_items = [
                            ("elapsed_time", f"{self.state.frame_counter//30}", "s", 30),
                            ("delta_angle", f"{current_angle:.2f}", "deg", 60),
                            ("circ_[x,y]", f"({x}, {y})", "pixels", 90),
                            ("cur_len(fitline)", f"{len(points)}", "", 150),
                            ("len(contours)", "Processing", "", 180),
                            ("len(results_list)", f"{len(self.state.registered_positions)}", "", 360),
                            ("results_list", f"{self.state.results[-5:] if self.state.results else []}", "", 390),
                            ("calibration", "READY" if self.state.calibration.is_calibrated else "NOT SET", "", 420)
                        ]
                        
                        self.visualizer.draw_info_overlay(black_foreground, info_items)
        else:
            self.state.circle_center = (x, y)
            #cv2.putText(black_foreground, "CALIBRATION MODE - Select MIN/MAX lines",  (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        self.visualizer.draw_calibration_lines(black_foreground, self.state.calibration)
        self.visualizer.draw_calibration_lines(frame, self.state.calibration)
        
        if self.state.calibration_mode:
            overlay = black_foreground.copy()
            cv2.rectangle(overlay, (0, 0), (black_foreground.shape[1], 60), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, black_foreground, 0.5, 0, black_foreground)
            
            #if self.state.selection_stage == 1:
            #    text = "CALIBRATION: Click to select MIN line (Green)"
            #    cv2.putText(black_foreground, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            #elif self.state.selection_stage == 2:
            #    text = "CALIBRATION: Click to select MAX line (Red)"
            #    cv2.putText(black_foreground, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return black_foreground
    def initialize_video(self):

        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def setup_windows(self):

        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self._mouse_callback)
        
        cv2.namedWindow('CONTROLS')
        cv2.createTrackbar('thresh_value',      'CONTROLS',         self.processing_params.thresh_value,        255, lambda v: setattr(self.processing_params, 'thresh_value', v))
        cv2.createTrackbar('thresh_value2',     'CONTROLS',         self.processing_params.thresh_value2,       255, lambda v: setattr(self.processing_params, 'thresh_value2', v))
        cv2.createTrackbar('H_varThreshold',    'CONTROLS',         self.processing_params.var_threshold,       255, lambda v: setattr(self.processing_params, 'var_threshold', v))
        cv2.createTrackbar('ROI Radius',        'CONTROLS',         self.roi_config.radius,                     300, lambda v: setattr(self.roi_config, 'radius', v))
        
        cv2.namedWindow('output_thresh')
        cv2.createTrackbar('ring_thresh_low',   'output_thresh',    self.processing_params.ring_thresh_low,     255, lambda v: setattr(self.processing_params, 'ring_thresh_low', v))
        cv2.createTrackbar('ring_thresh_high',  'output_thresh',    self.processing_params.ring_thresh_high,    255, lambda v: setattr(self.processing_params, 'ring_thresh_high', v))
        cv2.createTrackbar('N regions',         'output_thresh',    self.processing_params.n_regions,           100, lambda v: setattr(self.processing_params, 'n_regions', max(1, v)))
        
        cv2.namedWindow('detected_regions')

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state.calibration_mode:
                self.handle_calibration_click(x, y)
            else:
                self.roi_config.center_x = x
                self.roi_config.center_y = y
                print(f"Circle center set to ({x}, {y})")

    def register_position(self):
        if (self.state.black_foreground is not None and
            self.state.circle_center is not None and
            self.state.extended_point is not None):
            
            position = (self.state.circle_center, self.state.extended_point)
            self.state.registered_positions.append(position)
            self.state.last_registered_line = position
            self.calculate_all_angles()
            print(f"Position registered. Total positions: {len(self.state.registered_positions)}")

    def calculate_all_angles(self):
        if len(self.state.registered_positions) < 2:
            return
        
        angles = []
        first_line = self.state.registered_positions[0]
        
        for line in self.state.registered_positions:
            angle = self.angle_calculator.angle_between_lines(first_line, line)
            angles.append(angle)
        
        max_angle = max(angles) if angles else 0
        self.state.results = [self.angle_calculator.angle_to_value(a, max_angle) for a in angles]
        self.state.angles = angles
        print(f"Calculated {len(self.state.results)} angles: {self.state.results}")

    def run(self):
        self.initialize_video()
        self.setup_windows()

        while True:
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LANCZOS4)
                frame = self.calibrator.undistort(frame)
                original_frame = frame.copy()
            else:
                original_frame = self.frame if self.frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            
            result_frame = self.process_frame(original_frame)
            
            if result_frame is not None:
                cv2.imshow("app", result_frame)
            
            current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.visualizer.draw_progress(original_frame, current_frame, self.total_frames)
            cv2.imshow('frame', original_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nExiting application...")
                break
            elif key == 32:  # SPACE
                self.register_position()
            elif key == ord('c'):
                self.calculate_all_angles()
            elif key == ord('v'):
                print(f"\nResults: {self.state.results}")
            elif key == ord('s'):
                self.roi_config.center_x = self.roi_config.center_x
                self.roi_config.center_y = self.roi_config.center_y
                self.roi_config.radius = self.roi_config.radius
                cv2.setTrackbarPos('ROI Radius', 'CONTROLS', self.roi_config.radius)
                print("ROI reset to default")
            elif key == ord('p'):
                self.paused = not self.paused
                print(f"Video {'paused' if self.paused else 'resumed'}")
            elif key == ord('r'):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.state.frame_counter = 0
                print("Video reset to beginning")
            elif key == ord('m'): 
                self.enter_calibration_mode()
            elif key == 27:  
                if self.state.calibration_mode:
                    self.exit_calibration_mode()
                else:
                    print("\nExiting application...")
                    break
            
            self.state.frame_counter += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        plt.ioff()
        plt.close('all')
        print("\nApplication closed.")

# ===========
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    video_path = r'record 0 new.mp4'
    app = GaugeReaderApp(video_path)
    app.run()