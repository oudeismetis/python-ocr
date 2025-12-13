import os
import argparse
from datetime import datetime

from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2


def create_output_folder():
    """Create timestamped output folder for debugging images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"ocr_debug_output_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"{folder_name}/images", exist_ok=True)
    os.makedirs(f"{folder_name}/text", exist_ok=True)
    return folder_name


def detect_book_corners(img):
    """
    Detect the corners of a book in an image using contour detection.
    Returns the four corner points of the largest rectangular contour.
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Sort contours by area (largest first)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    book_corners = None
    
    # Look for the largest rectangular contour (likely the book)
    for contour in contours[:10]:  # Check top 10 largest contours
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # If we found a contour with 4 vertices, it's likely our book
        if len(approx) == 4:
            book_corners = approx.reshape(4, 2)
            break
    
    if book_corners is None:
        print("Warning: Could not detect book corners. Using full image.")
        h, w = gray.shape
        book_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    
    return book_corners


def order_corners(corners):
    """
    Order corners in a consistent manner: top-left, top-right, bottom-right, bottom-left
    """
    # Sort by y-coordinate (top points first)
    corners = corners[np.argsort(corners[:, 1])]
    
    # Get top two and bottom two points
    top_points = corners[:2]
    bottom_points = corners[2:]
    
    # Sort top points by x-coordinate (left first)
    top_points = top_points[np.argsort(top_points[:, 0])]
    # Sort bottom points by x-coordinate (right first for bottom-right, bottom-left order)
    bottom_points = bottom_points[np.argsort(bottom_points[:, 0])[::-1]]
    
    return np.array([top_points[0], top_points[1], bottom_points[0], bottom_points[1]], dtype=np.float32)


def correct_perspective(img, corners):
    """
    Apply perspective correction to straighten the book image.
    """
    # Order the corners
    ordered_corners = order_corners(corners)
    
    # Calculate the width and height of the corrected image
    width_top = np.linalg.norm(ordered_corners[1] - ordered_corners[0])
    width_bottom = np.linalg.norm(ordered_corners[2] - ordered_corners[3])
    width = int(max(width_top, width_bottom))
    
    height_left = np.linalg.norm(ordered_corners[3] - ordered_corners[0])
    height_right = np.linalg.norm(ordered_corners[2] - ordered_corners[1])
    height = int(max(height_left, height_right))
    
    # Define destination points for perspective correction
    dst_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype=np.float32)
    
    # Calculate perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dst_corners)
    
    # Apply perspective correction
    corrected = cv2.warpPerspective(img, matrix, (width, height))
    
    return corrected


def visualize_detection(img, corners, output_path=None):
    """
    Draw the detected corners on the image for visualization.
    """
    vis_img = img.copy()
    if len(vis_img.shape) == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
    
    # Draw the corners
    for i, corner in enumerate(corners):
        cv2.circle(vis_img, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
        cv2.putText(vis_img, str(i), tuple(corner.astype(int)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    # Draw lines connecting the corners
    cv2.polylines(vis_img, [corners.astype(int)], True, (0, 0, 255), 3)
    
    if output_path:
        cv2.imwrite(output_path, vis_img)
        print(f"Visualization saved to {output_path}")
    
    return vis_img


def main(img_name, visualize=False):
    """
    Main function to process the book image with corner detection and OCR.
    """
    output_folder = create_output_folder()
    output_folder = output_folder + "/images/"
    try:
        # Load the image
        # img_pil = Image.open(img_name)
        # img = np.array(img_pil)
        img = cv2.imread(img_name)
        cv2.imwrite(os.path.join(output_folder, "01_original.png"), img)
        
        print(f"Processing image: {img_name}")
        print(f"Image shape: {img.shape}")
        
        # Detect book corners
        print("Detecting book corners...")
        corners = detect_book_corners(img)
        print(f"Detected corners: {corners}")
        
        # Visualize detection if requested
        if visualize:
            vis_img = visualize_detection(img, corners, os.path.join(output_folder, "02_detection.jpg"))
        
        # Apply perspective correction
        print("Applying perspective correction...")
        corrected_img = correct_perspective(img, corners)
        
        # Save corrected image for inspection
        cv2.imwrite(os.path.join(output_folder, "03_corrected.png"), corrected_img)
        print(f"Corrected image saved as {img_name}_corrected.jpg")
        
        # Perform OCR on the corrected image
        print("Performing OCR...")
        corrected_pil = Image.fromarray(corrected_img)
        text = image_to_string(corrected_pil)
        
        if not text.strip():
            print("No text detected in corrected image.")
        else:
            print("Extracted text:")
            print("-" * 50)
            print(text)
            print("-" * 50)
        
        return text
        
    except Exception as e:
        print(f"Error processing image: {e}")
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from book images with corner detection")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Save visualization of detected corners")
    
    args = parser.parse_args()
    
    main(args.image, args.visualize)
