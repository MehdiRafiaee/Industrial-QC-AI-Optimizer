"""
Simple Defect Detection using OpenCV
- Reads an image from dataset
- Converts to grayscale
- Applies Laplacian filter to highlight edges/defects
- Displays original and processed images side by side
"""

import cv2
import os
import argparse

def detect_defect(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Laplacian filter to detect edges (potential defects)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)

    # Display images
    cv2.imshow("Original Image", img)
    cv2.imshow("Defect Highlight (Laplacian)", laplacian)
    print(f"Showing image: {os.path.basename(image_path)}")
    print("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple defect detector using OpenCV")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    
    detect_defect(args.image)
