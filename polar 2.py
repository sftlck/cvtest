import cv2
import numpy as np
import matplotlib.pyplot as plt


N = 30
num_bins = 360
threshold_value = 127
limit_chart = 15

def create_polar_histogram(image_path, threshold_value, num_bins):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    height, width = img.shape[:2] 
    size = min(height, width)
    
    if height != width:
        print(f"Image is not square ({width}x{height}). Cropping to {size}x{size}")
        start_x = (width - size) // 2
        start_y = (height - size) // 2
        img = img[start_y:start_y+size, start_x:start_x+size]
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    _, black_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY_INV)
    
    center_x, center_y = size // 2, size // 2
    max_radius = size // 2
    
    angle_step = 360 / num_bins
    angles = np.arange(0, 360, angle_step)
    
    sector_black_counts = np.zeros(num_bins)
    sector_total_pixels = np.zeros(num_bins)
    sector_black_percentages = np.zeros(num_bins)
    
    y_coords, x_coords = np.indices((size, size))
    
    dx = x_coords - center_x
    dy = y_coords - center_y
    
    angles_rad = np.arctan2(dy, dx)
    angles_deg = np.degrees(angles_rad)
    angles_deg = np.where(angles_deg < 0, angles_deg + 360, angles_deg)
    
    distances = np.sqrt(dx**2 + dy**2)
    
    circle_mask = distances <= max_radius
    
    for i, angle_start in enumerate(angles):
        angle_end = angle_start + angle_step
        
        if i == num_bins - 1:
            angle_mask = ((angles_deg >= angle_start) & (angles_deg <= 360)) | \
                         ((angles_deg >= 0) & (angles_deg < angle_step))
        else:
            angle_mask = (angles_deg >= angle_start) & (angles_deg < angle_end)
        
        sector_mask = angle_mask & circle_mask
        
        sector_total = np.sum(sector_mask)
        sector_black = np.sum(black_mask[sector_mask] > 0)
        
        sector_total_pixels[i] = sector_total
        sector_black_counts[i] = sector_black
        
        if sector_total > 0:
            sector_black_percentages[i] = (sector_black / sector_total) * 100
        else:
            sector_black_percentages[i] = 0

    fig = plt.figure(figsize=(5, 5))
    
    ax0 = plt.subplot(1,1,1, projection='polar')
    theta_rose = np.radians(np.append(angles, angles[0]))
    values_rose = np.append(sector_black_percentages, sector_black_percentages[0])
    
    ax0.fill(theta_rose, values_rose, alpha=0.5, color='black')
    ax0.plot(theta_rose, values_rose, color='black', linewidth=1)
    
    ax0.set_theta_zero_location('N')
    ax0.set_theta_direction(-1)
    ax0.set_ylim(0, limit_chart)
    ax0.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("="*60)
    
if __name__ == "__main__":
    #image_path = r'C:\Users\AdmPGE\Desktop\Castro\Viscomp\Imagens\teste15.png'
    image_path = r'C:\Users\AdmPGE\Desktop\Castro\Viscomp\Imagens\teste12.png'
    
    results = create_polar_histogram(image_path, threshold_value,num_bins)