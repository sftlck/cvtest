import cv2
import matplotlib.pyplot as plt
from subpixel_edges import subpixel_edges

image = r'C:\Users\Castro\Desktop\Computa\TestesVC\2026\01-26\teste2.jpg'

try:
    img = cv2.imread(image)
    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)

    hthreshold = 40
    lthreshold = 10

    edges = subpixel_edges(img_gray, hthreshold, 0, 100)  
    edges2 = subpixel_edges(img_gray, lthreshold, 0, 100) 

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img)
    axes[0].quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40, color='red')
    axes[0].set_title(f"Sub-pixel Edges (threshold=40)\nDetected edges: {len(edges.x)}")
    axes[0].axis('off')

    axes[1].imshow(img)
    axes[1].quiver(edges2.x, edges2.y, edges2.nx, -edges2.ny, scale=40, color='blue')
    axes[1].set_title(f"Sub-pixel Edges (threshold=10)\nDetected edges: {len(edges2.x)}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    print("\n--- Alternative: Separate figures ---")
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40, color='red')
    #plt.title(f"Sub-pixel Edges (threshold={lthreshold})\nDetected edges: {len(edges.x)}")
    plt.axis('off')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.imshow(img)
    plt.quiver(edges2.x, edges2.y, edges2.nx, -edges2.ny, scale=40, color='blue')
    plt.title(f'Sub-pixel Edges (threshold= {hthreshold} )\nDetected edges: {len(edges2.x)}')
    plt.axis('off')
    plt.show()

except FileNotFoundError:
    print(f"Error: '{image}' not found. Please provide an image file.")