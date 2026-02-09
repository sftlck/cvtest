import cv2
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import patches

N = 6
num_bins = 66
threshold_value = 127


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
    
    sorted_indices = np.argsort(sector_black_percentages)[::-1]
    top_5_indices = sorted_indices[:N]
    top_5_percentages = sector_black_percentages[top_5_indices]
    
    top_5_sectors = []
    for idx in top_5_indices:
        angle_start = angles[idx]
        angle_end = angle_start + angle_step
        mid_angle = angle_start + angle_step/2
        radius_point = 0.7 * max_radius
        
        angle_rad = np.radians(mid_angle)
        point_x = center_x + radius_point * np.cos(angle_rad)
        point_y = center_y + radius_point * np.sin(angle_rad)
        
        top_5_sectors.append({
            'sector_index': idx,
            'angle_start': angle_start,
            'angle_end': angle_end,
            'mid_angle': mid_angle,
            'black_percentage': sector_black_percentages[idx],
            'point_x': point_x,
            'point_y': point_y
        })
    
    fig = plt.figure(figsize=(18, 12))
    
    ax1 = plt.subplot(2, 3, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax1.imshow(img_rgb)
    
    circle = plt.Circle((center_x, center_y), max_radius, color='blue', 
                       fill=False, linewidth=1, alpha=0.7)
    ax1.add_patch(circle)
    
    ax1.plot(center_x, center_y, 'r+', markersize=7, markeredgewidth=2)
    
    colors = ['red', 'red', 'red', 'red', 'red']
    
    for i, sector in enumerate(top_5_sectors):
        start_angle_rad = np.radians(sector['angle_start'])
        end_angle_rad = np.radians(sector['angle_end'])
        
        ax1.plot([center_x, center_x + max_radius * np.cos(start_angle_rad)],
                [center_y, center_y + max_radius * np.sin(start_angle_rad)],
                color=colors[i % len(colors)], linewidth=1, alpha=0.7)
        ax1.plot([center_x, center_x + max_radius * np.cos(end_angle_rad)],
                [center_y, center_y + max_radius * np.sin(end_angle_rad)],
                color=colors[i % len(colors)], linewidth=1, alpha=0.7)
        
    for sector in top_5_sectors[:3]:
        arc = patches.Arc((center_x, center_y), 2*max_radius*0.3, 2*max_radius*0.3,
                     theta1=sector['angle_start'], theta2=sector['angle_end'],
                     color=colors[top_5_sectors.index(sector) % len(colors)], 
                     linewidth=3, alpha=0.5)
        ax1.add_patch(arc)
    
    ax1.set_title(f'original img spots', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.axis('off')
    
    
    ax4 = plt.subplot(2, 3, 4, projection='polar')
    theta_rose = np.radians(np.append(angles, angles[0]))
    values_rose = np.append(sector_black_percentages, sector_black_percentages[0])
    
    ax4.fill(theta_rose, values_rose, alpha=0.5, color='black')
    ax4.plot(theta_rose, values_rose, color='black', linewidth=2)
    
    ax4.set_theta_zero_location('N')
    ax4.set_theta_direction(-1)
    ax4.set_ylim(0, 10)
    ax4.grid(True, alpha=0.3)
    
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    top_5_data = []
    for i, sector in enumerate(top_5_sectors):
        top_5_data.append([
            sector['sector_index'] + 1,
            f"{sector['angle_start']:.0f}°-{sector['angle_end']:.0f}°",
            f"{sector['mid_angle']:.1f}°",
            f"{sector['black_percentage']:.1f}%",
            f"({sector['point_x']:.0f}, {sector['point_y']:.0f})"
        ])
    
    ax6 = plt.subplot(2, 3, 5)
    ax6.axis('off')
    
    total_pixels_circle = np.sum(circle_mask)
    total_black_circle = np.sum(black_mask[circle_mask] > 0)
    overall_black_percentage = (total_black_circle / total_pixels_circle * 100) if total_pixels_circle > 0 else 0
    
    max_sector_idx = np.argmax(sector_black_percentages)
    min_sector_idx = np.argmin(sector_black_percentages[sector_total_pixels > 0])
    
    stats_text = f"""
    Overall Statistics:
    --------------------
    Image Size: {size}×{size} pixels
    Circle Radius: {max_radius} pixels
    Pixels in Circle: {total_pixels_circle:,}
    Black Pixels: {total_black_circle:,}
    Black Percentage: {overall_black_percentage:.1f}%
    
    Sector Analysis ({num_bins} sectors):
    --------------------
    Max Black: Sector {max_sector_idx+1}
      Angle: {angles[max_sector_idx]:.0f}°-{angles[max_sector_idx]+angle_step:.0f}°
      Black: {sector_black_percentages[max_sector_idx]:.1f}%

    """
    
    for i, sector in enumerate(top_5_sectors):
        stats_text += f"\n  {i+1}. Sector {sector['sector_index']+1}: {sector['black_percentage']:.1f}%"
        stats_text += f" ({sector['angle_start']:.0f}°-{sector['angle_end']:.0f}°)"
    
    stats_text += f"""
    
    Threshold: {threshold_value}/255
    """
    
    ax6.text(0, 1, stats_text, fontfamily='monospace', fontsize=8,
             verticalalignment='top', transform=ax6.transAxes)
    ax6.set_title('Statistics', pad=20)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("="*60)
    for i, sector in enumerate(top_5_sectors):
        print(f"\n{i+1}. Sector {sector['sector_index']+1}:")
        print(f"   Angle Range: {sector['angle_start']:.0f}° to {sector['angle_end']:.0f}°")
        print(f"   Mid Angle: {sector['mid_angle']:.1f}°")
        print(f"   Black Percentage: {sector['black_percentage']:.2f}%")
        print(f"   Representative Point (x,y): ({sector['point_x']:.1f}, {sector['point_y']:.1f})")
        print(f"   Black Pixel Count: {sector_black_counts[sector['sector_index']]:.0f}")
        print(f"   Total Pixels in Sector: {sector_total_pixels[sector['sector_index']]:.0f}")
    
    return {
        'angles': angles,
        'black_percentages': sector_black_percentages,
        'black_counts': sector_black_counts,
        'total_counts': sector_total_pixels,
        'overall_percentage': overall_black_percentage,
        'center': (center_x, center_y),
        'radius': max_radius,
        'top_5_sectors': top_5_sectors
    }

if __name__ == "__main__":
    image_path = r'C:\Users\AdmPGE\Desktop\Castro\Viscomp\Imagens\teste12.png'
    
    results = create_polar_histogram(image_path, threshold_value,num_bins)