import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Set
import matplotlib.pyplot as plt
from enum import Enum

class CalibrationStage(Enum):

    NOT_CALIBRATING =       0
    SELECTING_MIN =         1
    SELECTING_MAX =         2

# =======================
# CLASSE PARA GLOBALIZAROS DADOS DO INDICADOR
# ============================================================================

@dataclass
class CameraParams:
    """Camera calibration parameters"""
    fx:                     float = 426.7231198260606
    fy:                     float = 424.44185129184774
    skew:                   float = 0.0
    cx:                     float = 308.3359965293331
    cy:                     float = 233.71108395163847
    width:                  int = 640
    height:                 int = 480
    r0:                     int = 2316661
    r1:                     int = -32512066

@dataclass
class ProcessingParams:
    """Image processing parameters"""
    n_regions:              int = 40
    min_region_area:        int = 10
    canny1:                 int = 10
    canny2:                 int = 255
    thresh_value:           int = 214
    thresh_value2:          int = 255
    dp:                     int = 7
    param1:                 int = 102
    param2:                 int = 306 
    min_radius:             int = 90
    max_radius:             int = 100
    distance:               int = 201
    history:                int = 300
    var_threshold:          int = 120
    gamma:                  float = 1.0
    alpha:                  float = 1.5
    beta:                   float = 0.0
    min_contour_area:       int = 15
    ring_scale_factor:      float = 0.8
    ring_thresh_low:        int = 170
    ring_thresh_high:       int = 255
    value_range:            float = 35.0
    resolution:             float = 0.1
    num_bins:               int = 360
    threshold_value:        int = 127
    limit_chart:            int = 25
    extension_beyond:       int = 15  
    
@dataclass
class ROIConfig:
    """Region of Interest configuration"""
    center_x:               int = 326
    center_y:               int = 267
    radius:                 int = 118
    selected:               bool = True

@dataclass
class LineData:
    """Represents a line from center to a detected point"""
    centroid:               Tuple[int, int]  
    center:                 Tuple[int, int]    
    angle_from_min:         float = 0.0  
    absolute_angle:         float = 0.0  
    circle_intersection:    Optional[Tuple[int, int]] = None  
    extended_end:           Optional[Tuple[int, int]] = None  
    is_tick_mark:           bool = False  
    index:                  int = -1  


@dataclass
class NeedleData:
    """Current needle position and derived data"""
    centroid:               Optional[Tuple[int, int]] = None  
    tip:                    Optional[Tuple[int, int]] = None  
    circle_intersection:    Optional[Tuple[int, int]] = None  
    absolute_angle:         float = 0.0  
    relative_angle:         float = 0.0  
    
    def as_line(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """Return as line tuple (center, tip) for angle calculations"""
        if self.tip:
            return (self.tip, self.centroid)  
        return None

@dataclass
class GaugeData:
    """Complete data structure for gauge state"""
    center:                 Tuple[int, int] = (0, 0)
    radius:                 int = 0
    circle_detected:        bool = False
    
    min_line:               Optional[LineData] = None
    max_line:               Optional[LineData] = None
    is_calibrated:          bool = False
    
    detected_lines:         List[LineData] = field(default_factory=list)
    tick_mark_angles:       List[float] = field(default_factory=list) 
    
    needle:                 NeedleData = field(default_factory=NeedleData)

    crossed_angles:         Dict[float, int] = field(default_factory=dict)
    max_crossed:            bool = False
    
    min_value:              float = 0.0
    max_value:              float = 100.0
    total_angle_span:       float = 0.0  
    
    def update_tick_marks(self):
        """Update tick mark list from detected lines"""
        self.tick_mark_angles = sorted([
            line.angle_from_min for line in self.detected_lines 
            if line.is_tick_mark
        ])
    
    def get_crossed_count(self) -> int:
        """Get number of crossed tick marks"""
        return len(self.crossed_angles)
    
    def get_progress(self) -> Tuple[int, int]:
        """Get (crossed, total) tick marks"""
        return (len(self.crossed_angles), len(self.tick_mark_angles))
    
    def reset_tracking(self):
        """Reset crossing tracking"""
        self.crossed_angles.clear()
        self.max_crossed = False
    
    def get_max_relative_angle(self) -> float:
        """Get MAX line angle relative to MIN"""
        if not self.min_line or not self.max_line:
            return 0.0
        return self.max_line.angle_from_min

# =======================
# CONFIGURAÇÃO
# ============================================================================


@dataclass
class CalibrationData:
    """Legacy class for backward compatibility"""
    min_line:       Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    max_line:       Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
    min_value:      float = 0.0
    max_value:      float = 100.0
    is_calibrated:  bool = False
    frozen_frame:   Optional[np.ndarray] = None

# ========================
# PROCESSING STATE (SIMPLIFIED)
# ============================================================================

class ProcessingState:
    """Global processing state - now much simpler!"""
    
    def __init__(self):
        self.frame_counter: int = 0
        self.calibration_mode: bool = False
        self.calibration_stage: CalibrationStage = CalibrationStage.NOT_CALIBRATING
        self.frozen_frame: Optional[np.ndarray] = None

# =========================================
# CAMERA CALIBRATOR (UNCHANGED)
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
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, 
            (self.params.width, self.params.height), 1, 
            (self.params.width, self.params.height)
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix, self.dist_coeffs, None, new_camera_matrix,
            (self.params.width, self.params.height), cv2.CV_16SC2
        )
        return map1, map2, roi, new_camera_matrix
    
    def undistort(self, frame: np.ndarray) -> np.ndarray:
        """Apply undistortion to frame"""
        undistorted = cv2.remap(frame, self.map1, self.map2, cv2.INTER_LANCZOS4)
        x, y, w, h = self.roi
        return undistorted[y:y+h, x:x+w]

# =========================+
# ANGLE CALCULATOR
# ============================================================================

class AngleCalculator:
    """All angle-related calculations in one place"""
    
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
        """Calculate absolute angle (0-360°) from vector components"""
        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        if angle_deg < 0:
            angle_deg += 360
        elif angle_deg == 0:
            angle_deg = 0.000001
        return angle_deg
    
    def calculate_line_angle(self, line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """Calculate absolute angle of a line defined as (point, center)"""
        point, center = line
        dx = point[0] - center[0]
        dy = point[1] - center[1]
        return self.calculate_angle_from_vector(dx, dy)
    
    @staticmethod
    def directed_angle(line1: Tuple[Tuple[int, int], Tuple[int, int]], 
                       line2: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """
        Calculate clockwise angle from line1 to line2 (0-360°)
        Lines are given as (point, center)
        """
        point1, center1 = line1
        point2, center2 = line2
        
        v1 = np.array([point1[0] - center1[0], point1[1] - center1[1]], dtype=np.float64)
        v2 = np.array([point2[0] - center2[0], point2[1] - center2[1]], dtype=np.float64)
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle_magnitude = np.degrees(np.arccos(dot))
        
        cross = v1_norm[0] * v2_norm[1] - v1_norm[1] * v2_norm[0]
        
        if cross < 0:
            return float(angle_magnitude)
        else:
            return float(360 - angle_magnitude)
    
    def relative_angle_from_min(self, line: Tuple[Tuple[int, int], Tuple[int, int]], min_line: Tuple[Tuple[int, int], Tuple[int, int]]) -> float:
        """Calculate clockwise angle from MIN line to given line"""
        return self.directed_angle(min_line, line)
    
    def angle_to_value(self, angle: float, max_angle: float) -> float:
        """Convert angle to measurement value"""
        if max_angle == 0:
            return 0
        return round(self.value_range * (angle / max_angle), self.res_calc)
    
    @staticmethod
    def find_line_circle_intersections(line_start: Tuple[int, int],  
                                       line_end: Tuple[int, int],  
                                       circle_center: Tuple[int, int],  
                                       circle_radius: int) -> List[Tuple[int, int]]:
        """Find intersection points between a line segment and a circle"""
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

# =========================+
# IMAGE PROCESSING UTILITIES
# ============================================================================

class ImageProcessor:
    """Core image processing utilities"""
    
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.app = app
        self.angle_calculator = AngleCalculator(params.value_range, params.resolution)
        
        self.last_component_centroids = []
        self.last_offset = None
    
    def analyze_ring(self, frame: np.ndarray, offset: Optional[Tuple[int, int]], 
                    gauge_data: GaugeData, result_frame: np.ndarray, 
                    original_frame: np.ndarray) -> List[LineData]:
        """
        Analyze ring area to detect all lines from center to black regions
        Returns list of LineData objects
        """
        img_copy = frame.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        _, black_mask = cv2.threshold(gray, self.params.threshold_value, 255, cv2.THRESH_BINARY_INV)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(black_mask, connectivity=8)
        
        detected_lines = []
        component_centroids = []
        
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            
            if area >= 5:  # Minimum area threshold
                x = stats[label, cv2.CC_STAT_LEFT]
                y = stats[label, cv2.CC_STAT_TOP]
                w = stats[label, cv2.CC_STAT_WIDTH]
                h = stats[label, cv2.CC_STAT_HEIGHT]
                
                centroid = (int(centroids[label][0]), int(centroids[label][1]))
                component_centroids.append(centroid)
                
                if offset:
                    original_centroid = (offset[0] + centroid[0], offset[1] + centroid[1])
                else:
                    original_centroid = centroid
                
                line = LineData(
                    centroid=original_centroid,
                    center=gauge_data.center,
                    absolute_angle=self.angle_calculator.calculate_angle_from_vector(
                        original_centroid[0] - gauge_data.center[0],
                        original_centroid[1] - gauge_data.center[1]
                    )
                )
                
                self._calculate_line_intersection(line, gauge_data)
                
                detected_lines.append(line)
        
        self.last_component_centroids = component_centroids
        self.last_offset = offset
        
        self._draw_debug_regions(img_copy, component_centroids, stats)
        
        return detected_lines
    
    def _calculate_line_intersection(self, line: LineData, gauge_data: GaugeData):
        """Calculate where the line from center through centroid meets the circle"""
        dx = line.centroid[0] - line.center[0]
        dy = line.centroid[1] - line.center[1]
        dist = math.sqrt(dx*dx + dy*dy)
        
        if dist > 0:
            dx_norm = dx / dist
            dy_norm = dy / dist
            
            circle_x = int(line.center[0] + dx_norm * gauge_data.radius)
            circle_y = int(line.center[1] + dy_norm * gauge_data.radius)
            line.circle_intersection = (circle_x, circle_y)
            
            line.extended_end = ( int(circle_x + dx_norm * self.params.extension_beyond), int(circle_y + dy_norm * self.params.extension_beyond) )
            line.extended_end_text = ( int(circle_x + dx_norm * self.params.extension_beyond + self.params.extension_beyond * .2), int(circle_y + dy_norm * self.params.extension_beyond + self.params.extension_beyond * .2) )
    
    def _draw_debug_regions(self, img: np.ndarray, centroids: List[Tuple[int, int]], stats: np.ndarray):
        if len(centroids) > 0:
            display_img = img.copy()
            for i, region in enumerate(stats[1:51]):  # First 50 regions
                x, y, w, h = region[cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]
                cv2.rectangle(display_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cv2.imshow('detected_regions', display_img)
    
    def draw_lines_on_frame(self, result_frame: np.ndarray, original_frame: np.ndarray, lines: List[LineData], gauge_data: GaugeData):
        for i, line in enumerate(lines):
            cv2.line(result_frame, line.center, line.centroid, (90, 90, 90), 1)
            cv2.line(original_frame, line.center, line.centroid, (0, 0, 125), 1)
            
            if line.is_tick_mark and line.extended_end:
                cv2.line(result_frame, line.center, line.extended_end, (200, 200, 200), 1)
                
                if line.circle_intersection:
                    cv2.circle(result_frame, line.circle_intersection, 3, (255, 255, 0), -1)
                
                cv2.putText(result_frame, str(int(line.angle_from_min)), line.extended_end_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1) #### ARMAZENA O ÂNGULO A SER IMPRESSO
    
    #def draw_lines_on_frame(self, result_frame: np.ndarray, original_frame: np.ndarray, lines: List[LineData], gauge_data: GaugeData):
    #    cv2.putText(result_frame, text, , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    def find_nearest_region(self, click_x: int, click_y: int, 
                           component_centroids: List[Tuple[int, int]], 
                           offset: Tuple[int, int], 
                           max_distance: int = 50) -> Optional[Tuple[int, int]]:
        if not component_centroids or offset is None:
            return None
        
        crop_click_x, crop_click_y = click_x - offset[0], click_y - offset[1]
        
        min_dist = float('inf')
        closest_centroid = None
        
        for cx, cy in component_centroids:
            dist = np.sqrt((cx - crop_click_x)**2 + (cy - crop_click_y)**2)
            if dist < min_dist and dist < max_distance:
                min_dist = dist
                closest_centroid = (cx, cy)
        
        return closest_centroid
    
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

    def process(self, ring_img: np.ndarray, result_frame=None, offset=None) -> List[LineData]:
        """
        Process ring image and return detected lines
        """
        gray = self.image_processor.convert_to_gray(ring_img)
        _, output_thresh = cv2.threshold(
            gray, self.params.ring_thresh_low,
            self.params.ring_thresh_high, cv2.THRESH_BINARY
        )
        
        # Analyze ring to get lines
        lines = self.image_processor.analyze_ring(
            ring_img, offset, self.app.gauge_data, result_frame, self.app.frame
        )
        
        return lines

# ======
# NEEDLE DETECTOR
# ============================================================================

class NeedleDetector:
    def __init__(self, params: ProcessingParams, app: 'GaugeReaderApp' = None):
        self.params = params
        self.app = app
        self.ring_processor = RingProcessor(params, self)
        self.backSub = cv2.createBackgroundSubtractorMOG2(
            history=params.history,
            varThreshold=params.var_threshold,
            detectShadows=True
        )
    
    def detect_circle(self, gray: np.ndarray, roi_config: Optional[ROIConfig] = None) -> Optional[Tuple[int, int, int]]:
        """Detect circular gauge using Hough transform or ROI"""
        if roi_config and roi_config.selected:
            return (roi_config.center_x, roi_config.center_y, roi_config.radius)
        
        circles = cv2.HoughCircles(
            gray, cv2.HOUGH_GRADIENT, self.params.dp, self.params.distance,
            param1=self.params.param1, param2=self.params.param2,
            minRadius=self.params.min_radius, maxRadius=self.params.max_radius
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            x, y, r = circles[0]
            return (x, y, int(r * 0.9))
        
        return None
    
    def extract_needle(self, frame: np.ndarray, circle_data: Tuple[int, int, int],
                      mask: np.ndarray, gauge_data: GaugeData) -> Optional[NeedleData]:
        """Extract needle from frame and update gauge_data"""
        x, y, r = circle_data
        
        fg_mask_full = self.backSub.apply(frame)
        
        cv2.circle(mask, (x, y), r, 255, -1)
        fg_mask_circular = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)
        
        _, mask_thresh = cv2.threshold(
            fg_mask_circular, self.params.thresh_value,
            self.params.thresh_value2, cv2.THRESH_BINARY
        )
        
        cv2.imshow('mask_thresh', mask_thresh)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        mask_eroded = cv2.morphologyEx(mask_thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        large_contours = [
            cnt for cnt in contours 
            if cv2.contourArea(cnt) > self.params.min_contour_area
        ]
        
        if not large_contours:
            return None
        
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
            if (py < rectangle_mask.shape[0] and px < rectangle_mask.shape[1] and 
                rectangle_mask[py, px] > 0 and dist <= r):
                all_points.append([px, py])
        
        if len(all_points) < 20:
            return None
        
        fitline = np.array(all_points, dtype=np.float32)
        [vx, vy, x0, y0] = cv2.fitLine(fitline, cv2.DIST_L2, 0, 0.01, 0.01)
        
        if abs(vx[0]) <= 1e-6:
            return None
        
        centroid = (int(x0[0]), int(y0[0]))
        
        dx = centroid[0] - x
        dy = centroid[1] - y
        distance = np.sqrt(dx*dx + dy*dy)
        
        if distance <= 0:
            return None
        
        dx_norm = dx / distance
        dy_norm = dy / distance
        
        extension = r - distance + 0.1 * r
        tip_x = int(x + dx_norm * (distance + extension))
        tip_y = int(y + dy_norm * (distance + extension))
        
        intersection_x = int(x + dx_norm * r)
        intersection_y = int(y + dy_norm * r)
        
        needle = NeedleData(
            centroid=centroid,
            tip=(tip_x, tip_y),
            circle_intersection=(intersection_x, intersection_y),
            absolute_angle=self.app.angle_calculator.calculate_angle_from_vector(tip_x - x, tip_y - y))
        
        if gauge_data.min_line:
            needle_line = (needle.tip, (x, y))
            min_line = (gauge_data.min_line.centroid, gauge_data.min_line.center)
            needle.relative_angle = self.app.angle_calculator.relative_angle_from_min(min_line,needle_line)
        
        return needle
    
    def process_ring(self, frame, circle_data, result_frame=None) -> List[LineData]:
        """Process ring area and return detected lines"""
        x, y, r = circle_data
        inner_radius = int(r * self.params.ring_scale_factor)
        
        if inner_radius < r:
            ring_cropped, offset = self.ring_processor.image_processor.crop_ring_area(frame, x, y, inner_radius, r)
            
            if ring_cropped is not None and ring_cropped.size > 0:
                return self.ring_processor.process(ring_cropped, result_frame, offset)
        
        return []

# ===========
# VISUALIZATION
# ============================================================================

class Visualizer:
    """Handle all drawing operations"""
    
    @staticmethod
    def draw_circle(frame: np.ndarray, center: Tuple[int, int], radius: int, color: Tuple[int, int, int] = (0, 255, 0)):
        cv2.circle(frame, center, radius, color, 1)
        cv2.circle(frame, center, 2, color, 1)
    
    @staticmethod
    def draw_needle(frame: np.ndarray, needle: NeedleData, color: Tuple[int, int, int] = (0, 255, 255)):
        if needle.tip and needle.centroid:
            cv2.line(frame, needle.centroid, needle.tip, color, 2)
            cv2.circle(frame, needle.centroid, 5, (0, 255, 0), 1)
    
    @staticmethod
    def draw_intersection_marker(frame: np.ndarray, point: Tuple[int, int], color: Tuple[int, int, int] = (0, 0, 255)):
        if point:
            cv2.drawMarker(frame, point, color, cv2.MARKER_SQUARE, 20, 1, 1)
    
    @staticmethod
    def draw_calibration_lines(frame: np.ndarray, gauge_data: GaugeData):
        """Draw MIN (green) and MAX (red) calibration lines"""
        if gauge_data.min_line:
            cv2.line(frame, gauge_data.min_line.center, gauge_data.min_line.centroid, (0, 255, 0), 1)
            cv2.circle(frame, gauge_data.min_line.centroid, 5, (0, 255, 0), -1)
        
        if gauge_data.max_line:
            cv2.line(frame, gauge_data.max_line.center, gauge_data.max_line.centroid, (0, 0, 255), 1)
            cv2.circle(frame, gauge_data.max_line.centroid, 5, (0, 0, 255), -1)
    
    @staticmethod
    def draw_info_overlay(frame: np.ndarray, gauge_data: GaugeData, state: ProcessingState):
        """Draw information overlay with all gauge data"""
        y_pos = 30
        info_items = [
            ("frame", f"{state.frame_counter}", "", y_pos),
            ("angle", f"{gauge_data.needle.absolute_angle:.1f}", "°", y_pos + 30),
        ]
        
        if gauge_data.min_line:
            info_items.append(("abs(rel angle)", f"{gauge_data.needle.relative_angle:.1f}", "[deg]", y_pos + 60))
        
        crossed, total = gauge_data.get_progress()
        info_items.append(("progress", f"{crossed}/{total}", "marks", y_pos + 90))
        
        if gauge_data.max_crossed:
            info_items.append(("MAX", "✓", "", y_pos + 120))
        
        for label, value, unit, y in info_items:
            text = f"{label}: {value}"
            if unit:
                text += f" [{unit}]"
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_DUPLEX, 0.4, (255, 255, 255), 1)
    
    @staticmethod
    def draw_calibration_mode(frame: np.ndarray, stage: CalibrationStage):
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        if stage == CalibrationStage.SELECTING_MIN:
            cv2.putText(frame, "min_flag", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        elif stage == CalibrationStage.SELECTING_MAX:
            cv2.putText(frame, "max_flag", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    @staticmethod
    def draw_progress(frame: np.ndarray, current: int, total: int):
        progress = f"Frame: {current}/{total} ({(current/total*100):.1f}%)"
        cv2.putText(frame, progress, (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# ==============
# MAIN APPLICATION
# ============================================================================

class GaugeReaderApp:
    def __init__(self, video_path: str):
        self.video_path =           video_path
        self.params =               ProcessingParams()
        self.roi_config =           ROIConfig()
        self.state =                ProcessingState()

        self.calibrator =           CameraCalibrator(CameraParams())
        self.angle_calculator =     AngleCalculator(self.params.value_range, self.params.resolution)
        self.image_processor =      ImageProcessor(self.params, app=self)
        self.needle_detector =      NeedleDetector(self.params, app=self)
        self.visualizer =           Visualizer()

        self.gauge_data =           GaugeData()

        self.cap =                  None
        self.total_frames =         0
        self.fps =                  0
        self.paused =               False
        
        self.current_result_frame = None
        self.frame =                None
        self.circle_data =          None

        self.last_needle_angle = 0

    def process_frozen_frame_for_calibration(self):
        """Process frozen frame to get lines for calibration"""
        if self.state.frozen_frame is None:
            print("No frozen frame available")
            return False
        
        gray = self.image_processor.convert_to_gray(self.state.frozen_frame)
        circle_data = self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            print("Could not detect circle in frozen frame")
            return False
        
        x, y, r = circle_data
        self.gauge_data.center = (x, y)
        self.gauge_data.radius = r
        self.gauge_data.circle_detected = True
        
        temp_result = np.zeros_like(self.state.frozen_frame)
        
        inner_radius = int(r * self.params.ring_scale_factor)
        if inner_radius < r:
            ring_cropped, offset = self.image_processor.crop_ring_area(
                self.state.frozen_frame, x, y, inner_radius, r
            )
            
            if ring_cropped is not None and ring_cropped.size > 0:
                lines = self.image_processor.analyze_ring(ring_cropped, offset, self.gauge_data, temp_result, self.state.frozen_frame)
                self.gauge_data.detected_lines = lines
                
                print(f"Found {len(lines)} regions in frozen frame")
                return len(lines) > 0
        
        return False

    def enter_calibration_mode(self):
        """Freeze frame and enter calibration mode"""
        if self.frame is not None:
            self.state.frozen_frame = self.frame.copy()
            self.state.calibration_mode = True
            self.state.calibration_stage = CalibrationStage.SELECTING_MIN
            self.paused = True
            
            success = self.process_frozen_frame_for_calibration()
            
    def exit_calibration_mode(self):
        """Exit calibration mode without saving"""
        self.state.calibration_mode = False
        self.state.calibration_stage = CalibrationStage.NOT_CALIBRATING
        self.paused = False
        print("Calibration cancelled")

    def handle_calibration_click(self, x: int, y: int):
        """Handle mouse clicks during calibration"""
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
        
        target_line = None
        for line in self.gauge_data.detected_lines:
            if (abs(line.centroid[0] - original_x) < 5 and 
                abs(line.centroid[1] - original_y) < 5):
                target_line = line
                break
        
        if not target_line:
            print("Could not find matching line data")
            return
        
        if self.state.calibration_stage == CalibrationStage.SELECTING_MIN:
            self.gauge_data.min_line = target_line
            target_line.is_tick_mark = True
            target_line.angle_from_min = 0 
            self.state.calibration_stage = CalibrationStage.SELECTING_MAX
            print(f"min_line: ({original_x}, {original_y})")
            
        elif self.state.calibration_stage == CalibrationStage.SELECTING_MAX:
            self.gauge_data.max_line = target_line
            target_line.is_tick_mark = True
            
            if self.gauge_data.min_line:
                min_line_tuple = (self.gauge_data.min_line.centroid, self.gauge_data.min_line.center)
                max_line_tuple = (target_line.centroid, target_line.center)
                max_angle = self.angle_calculator.relative_angle_from_min(max_line_tuple, min_line_tuple)
                target_line.angle_from_min = max_angle
                self.gauge_data.total_angle_span = max_angle
            
            self.gauge_data.is_calibrated = True
            self.state.calibration_mode = False
            self.state.calibration_stage = CalibrationStage.NOT_CALIBRATING
            self.paused = False
            
            self._update_all_tick_marks()
            
            print(f"max_line ({original_x}, {original_y})")
            print(f"angle_span: {self.gauge_data.total_angle_span:.1f}°")
    
    def _update_all_tick_marks(self):

        if not self.gauge_data.min_line:
            return
        
        min_line_tuple = (self.gauge_data.min_line.centroid, self.gauge_data.min_line.center)
        
        for line in self.gauge_data.detected_lines:
            line_tuple = (line.centroid, line.center)
            line.angle_from_min = self.angle_calculator.relative_angle_from_min(min_line_tuple,line_tuple)
            
            if self.gauge_data.max_line:
                max_angle = self.gauge_data.max_line.angle_from_min
                max_range = max_angle 
                line.is_tick_mark = (3 <= line.angle_from_min <= max_range + 3)
        
        self.gauge_data.update_tick_marks()
    
    def process_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Process a single frame"""
        enhanced_frame = self.image_processor.enhance_image(frame, self.params.alpha, self.params.beta)
        gray = self.image_processor.convert_to_gray(enhanced_frame)
        
        black_foreground = np.zeros_like(frame)
        self.current_result_frame = black_foreground
        self.frame = frame
        
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        
        circle_data = self.needle_detector.detect_circle(gray, self.roi_config)
        
        if not circle_data:
            return None
        
        x, y, r = circle_data
        self.gauge_data.center = (x, y)
        self.gauge_data.radius = r
        self.gauge_data.circle_detected = True
        
        self.visualizer.draw_circle(frame, (x, y), r)
        self.visualizer.draw_circle(black_foreground, (x, y), r)
        
        n_regions = cv2.getTrackbarPos('N regions', 'output_thresh')
        detected_lines = self.needle_detector.process_ring(enhanced_frame, circle_data, black_foreground)
        
        if detected_lines:
            self.gauge_data.detected_lines = detected_lines
            
            if self.gauge_data.is_calibrated:
                self._update_all_tick_marks()
            
            self.image_processor.draw_lines_on_frame(black_foreground, frame, detected_lines, self.gauge_data)
        
        if not self.state.calibration_mode:
            needle = self.needle_detector.extract_needle(enhanced_frame, circle_data, mask, self.gauge_data)
            
            if needle:
                self.gauge_data.needle = needle
                
                self.visualizer.draw_needle(black_foreground, needle)
                self.visualizer.draw_intersection_marker(black_foreground, needle.circle_intersection)
                cv2.putText(black_foreground, f"current needle: {needle.relative_angle:.1f} [deg]", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                if self.gauge_data.is_calibrated and self.gauge_data.tick_mark_angles:
                    
                    for tick_angle in self.gauge_data.tick_mark_angles:
                        #print(len(self.gauge_data.tick_mark_angles))
                        if needle.relative_angle > tick_angle:
                            
                            if not hasattr(self, 'last_needle_angle'):
                                self.last_needle_angle = 0
                            
                            if self.last_needle_angle <= tick_angle:
                                
                                if tick_angle not in self.gauge_data.crossed_angles:
                                    self.gauge_data.crossed_angles[tick_angle] = 1
                                else:
                                    self.gauge_data.crossed_angles[tick_angle] += 1
                                
                                #pass_count = self.gauge_data.crossed_angles[tick_angle]                            #### FINALMENTE, AGORA CONTA APENAS QUANDO PASSAR
                                #direction = "↑" if needle.relative_angle > self.last_needle_angle else "↓"
                                #print(f"PASS #{pass_count} {direction}: {int(tick_angle)}° tick (needle: {needle.relative_angle:.1f}°)")
                                print(f"{int(tick_angle)}° tick / needle: {needle.relative_angle:0f}°")            #### SÓ NÃO PODE PASSAR RÁPIDO DEMAIS
                                ly = 25
                                res = 2
                                for item in self.gauge_data.tick_mark_angles:
                                    if item < needle.relative_angle:
                                        cv2.putText(black_foreground, f"{int(item)}",      (black_foreground.shape[:2][1]-110, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
                                        cv2.putText(black_foreground, f"{res} [mca]",      (black_foreground.shape[:2][1]-75, ly),  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
                                        res += 2
                                        ly  += 15

                                    if (tick_angle + 4) > max(self.gauge_data.tick_mark_angles):
                                        print("="*50)
                                        print("MAX_ANGLE")
                                        print("="*50)
                                
                                    cv2.putText(black_foreground, f"current tick_angle: {tick_angle:.0f} [deg]", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200, 0), 1)
                    #cv2.putText(black_foreground, f"tick angles: ", (10, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)
                     
                    self.last_needle_angle = needle.relative_angle
                    
                if self.gauge_data.is_calibrated and self.gauge_data.max_line:
                    max_angle = self.gauge_data.max_line.angle_from_min
                    #if (needle.relative_angle > max_angle and not self.gauge_data.max_crossed):
                                        
                    self.gauge_data.max_crossed = True
        
        self.visualizer.draw_calibration_lines(black_foreground, self.gauge_data)
        self.visualizer.draw_calibration_lines(frame, self.gauge_data)
        
        if self.state.calibration_mode:
            self.visualizer.draw_calibration_mode(black_foreground, self.state.calibration_stage)
        
        self.visualizer.draw_info_overlay(black_foreground, self.gauge_data, self.state)
        
        return black_foreground
    
    def initialize_video(self):
        """Initialize video capture"""
        self.cap = cv2.VideoCapture(self.video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def setup_windows(self):
        """Setup OpenCV windows and trackbars"""
        cv2.namedWindow('frame')
        cv2.setMouseCallback('frame', self._mouse_callback)
        
        cv2.namedWindow('CONTROLS')
        cv2.createTrackbar('thresh_value',      'CONTROLS', self.params.thresh_value,           255, lambda v: setattr(self.params, 'thresh_value', v))
        cv2.createTrackbar('thresh_value2',     'CONTROLS', self.params.thresh_value2,          255, lambda v: setattr(self.params, 'thresh_value2', v))
        cv2.createTrackbar('H_varThreshold',    'CONTROLS', self.params.var_threshold,          255, lambda v: setattr(self.params, 'var_threshold', v))
        cv2.createTrackbar('ROI Radius',        'CONTROLS', self.roi_config.radius,             300, lambda v: setattr(self.roi_config, 'radius', v))
        
        cv2.namedWindow('output_thresh')
        cv2.createTrackbar('ring_thresh_low',   'output_thresh', self.params.ring_thresh_low,   255, lambda v: setattr(self.params, 'ring_thresh_low', v))
        cv2.createTrackbar('ring_thresh_high',  'output_thresh', self.params.ring_thresh_high,  255, lambda v: setattr(self.params, 'ring_thresh_high', v))
        cv2.createTrackbar('N regions',         'output_thresh', self.params.n_regions,         100, lambda v: setattr(self.params, 'n_regions', max(1, v)))
        
        cv2.namedWindow('detected_regions')

    def _mouse_callback(self, event, x, y, flags, param):
        """Mouse callback for calibration and ROI"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.state.calibration_mode:
                self.handle_calibration_click(x, y)
            else:
                self.roi_config.center_x = x
                self.roi_config.center_y = y
                print(f"Circle center set to ({x}, {y})")

    def register_position(self):
        """Legacy method - kept for compatibility"""
        pass

    def calculate_all_angles(self):
        """Legacy method - kept for compatibility"""
        pass

    def run(self):
        """Main application loop"""
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
            elif key == 32:  
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
                self.gauge_data.reset_tracking()
                print("Video and tracking reset")
            elif key == ord('m'): 
                self.enter_calibration_mode()
            elif key == ord('x'):  
                self.gauge_data.reset_tracking()
                print("Crossing trackers reset")
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