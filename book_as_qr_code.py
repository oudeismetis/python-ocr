import os
import sys
from datetime import datetime
import tempfile

from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2
import math

def create_output_folder():
    """Create timestamped output folder for debugging images"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"blur_debug_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    os.makedirs(f"output/{folder_name}/images", exist_ok=True)
    os.makedirs(f"output/{folder_name}/text", exist_ok=True)
    return folder_name

def detect_and_correct_skew(image):
    """Detect and correct text skew using Hough line transform"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is not None:
        angles = []
        for line in lines[:10]:  # Use first 10 lines
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            # Convert to skew angle (-45 to 45 degrees)
            if angle > 90:
                angle = angle - 180
            elif angle > 45:
                angle = angle - 90
            angles.append(angle)
        
        # Get median angle to avoid outliers
        if angles:
            skew_angle = np.median(angles)
            
            # Only correct if skew is significant (> 0.5 degrees)
            if abs(skew_angle) > 0.5:
                # Rotate image to correct skew
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
                corrected = cv2.warpAffine(image, rotation_matrix, (w, h), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
                return corrected
    
    return image

def detect_pages(image):
    """Detect individual pages in the image using contour detection"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and aspect ratio
    min_area = image.shape[0] * image.shape[1] * 0.1  # At least 10% of image
    page_contours = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Typical page aspect ratio is between 0.5 and 2.0
            if 0.5 < aspect_ratio < 2.0:
                page_contours.append((x, y, w, h))
    
    # Sort by x-coordinate (left to right)
    page_contours.sort(key=lambda x: x[0])
    
    return page_contours

def detect_book_text_areas(image):
    """Detect book pages and text areas using edge detection and contours"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Apply bilateral filter to reduce noise while keeping edges sharp
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply edge detection
    edges = cv2.Canny(filtered, 50, 150, apertureSize=3)
    
    # Dilate edges to connect nearby edge segments
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    
    # Find contours
    contours_result = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours_result) == 3:
        _, contours, _ = contours_result
    else:
        contours, _ = contours_result
    
    # Find the largest rectangular contours (likely book pages)
    page_candidates = []
    min_area = image.shape[0] * image.shape[1] * 0.05  # At least 5% of image
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            
            # Look for rectangular shapes that could be pages
            if len(approx) >= 4 and 0.3 < aspect_ratio < 3.0:
                page_candidates.append((x, y, w, h, area))
    
    # Sort by area (largest first) and take top candidates
    page_candidates.sort(key=lambda x: x[4], reverse=True)
    
    # Return bounding boxes of detected pages
    return [(x, y, w, h) for x, y, w, h, area in page_candidates[:2]]

def detect_book_text_areas_2(image):
    """Detect text areas using MSER (text blob detection)"""
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Create MSER detector
    mser = cv2.MSER_create()
    regions, _ = mser.detectRegions(gray)
    
    if not regions:
        # Fallback: return whole image
        return [(0, 0, image.shape[1], image.shape[0])]
    
    # Get bounding boxes of all text regions
    boxes = []
    for region in regions:
        x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
        boxes.append((x, y, w, h))
    
    # Merge overlapping boxes and find largest text areas
    merged_boxes = []
    min_area = 1000  # Minimum text area size
    
    for x, y, w, h in boxes:
        if w * h > min_area:
            merged_boxes.append((x, y, w, h))
    
    # If no good regions found, return whole image
    if not merged_boxes:
        return [(0, 0, image.shape[1], image.shape[0])]
    
    return merged_boxes[:2]  # Return top 2 regions

def perspective_correction(image, contour_points):
    """Apply perspective correction using detected corners"""
    # Order points: top-left, top-right, bottom-right, bottom-left
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]  # top-left
        rect[2] = pts[np.argmax(s)]  # bottom-right
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]  # top-right
        rect[3] = pts[np.argmax(diff)]  # bottom-left
        return rect
    
    # Get the ordered points
    rect = order_points(contour_points)
    (tl, tr, br, bl) = rect
    
    # Compute width and height of new image
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Destination points for perspective transform
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    
    # Compute perspective transform matrix and apply it
    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    
    return warped

def enhance_text_contrast(image):
    """Enhance text contrast using CLAHE and morphological operations"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Apply morphological operations to clean up noise
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def preprocess_book_image(img_name, output_folder):
    """Complete preprocessing pipeline for book images"""
    img_orig = cv2.imread(img_name)
    if img_orig is None:
        img_orig = np.array(Image.open(img_name))
        if len(img_orig.shape) == 3 and img_orig.shape[2] == 3:
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, "01_original.png"), img_orig)
    
    # Step 1: Detect and correct overall skew
    # skew_corrected = detect_and_correct_skew(img_orig)
    # cv2.imwrite(os.path.join(output_folder, "02_skew_corrected.png"), skew_corrected)
    
    skew_corrected = img_orig
    # Step 2: Detect individual pages
    # page_regions = detect_pages(skew_corrected)
    page_regions = detect_book_text_areas_2(skew_corrected)
    print(f"Found page regions: {page_regions}")
    
    results = []
    
    if page_regions:
        # Process each detected page separately
        for i, (x, y, w, h) in enumerate(page_regions):
            print(f"Working on page {i+1}")
            # Extract page region with some padding
            padding = 10
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(skew_corrected.shape[1], x + w + padding)
            y2 = min(skew_corrected.shape[0], y + h + padding)
            
            page_img = skew_corrected[y1:y2, x1:x2]
            cv2.imwrite(os.path.join(output_folder, f"03_page_{i+1}_extracted.png"), page_img)
            
            # Step 3: Apply additional skew correction to individual page
            page_deskewed = detect_and_correct_skew(page_img)
            cv2.imwrite(os.path.join(output_folder, f"04_page_{i+1}_deskewed.png"), page_deskewed)
            
            # Step 4: Enhance text contrast
            # enhanced_page = enhance_text_contrast(page_deskewed)
            # cv2.imwrite(os.path.join(output_folder, f"05_page_{i+1}_enhanced.png"), enhanced_page)
            
            # Step 5: Final cleanup
            # final_img = cv2.medianBlur(enhanced_page, 3)
            # cv2.imwrite(os.path.join(output_folder, f"06_page_{i+1}_final.png"), final_img)
            # results.append(final_img)
            results.append(page_deskewed)
    else:
        # If no pages detected, process entire image
        enhanced = enhance_text_contrast(skew_corrected)
        cv2.imwrite(os.path.join(output_folder, "03_single_page_enhanced.png"), enhanced)
        final_img = cv2.medianBlur(enhanced, 3)
        cv2.imwrite(os.path.join(output_folder, "04_single_page_final.png"), final_img)
        results.append(final_img)
    return results

def high_blur(img_name, output_folder):
    r"Complete preprocessing pipeline for book images"""
    img_orig = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img_orig is None:
        img_orig = np.array(Image.open(img_name))
        if len(img_orig.shape) == 3 and img_orig.shape[2] == 3:
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_folder, "01_original.png"), img_orig)

    blurred = cv2.GaussianBlur(img_orig, (25, 25), sigmaX=15, sigmaY=15)
    cv2.imwrite(os.path.join(output_folder, "02_blurred.png"), blurred)

    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY) if len(blurred.shape) == 3 else blurred
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite(os.path.join(output_folder, "03_binary.png"), thresh)

def load_image(img_name):
    img_orig = cv2.imread(img_name, cv2.IMREAD_UNCHANGED)
    if img_orig is None:
        img_orig = np.array(Image.open(img_name))
        if len(img_orig.shape) == 3 and img_orig.shape[2] == 3:
            img_orig = cv2.cvtColor(img_orig, cv2.COLOR_RGB2BGR)
    return img_orig

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.equalizeHist(gray)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    #     35, 15
    #     11, 2
    thresh = cv2.bitwise_not(thresh)
    return thresh

def morph_cleanup(img):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    clean = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel)
    return clean

def find_bright_blobs(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_like = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 0.05 * img.shape[0] * img.shape[1]:
            continue  # skip small bright patches
        rect = cv2.minAreaRect(cnt)
        box = np.intp(cv2.boxPoints(rect))
        page_like.append((box, rect))
    return page_like

def find_pages(img, page_like):
    if len(page_like) == 1:
        (box, rect) = page_like[0]
        x_profile = np.mean(img, axis=0)
        mid = np.argmin(x_profile)  # likely the book spine

        left_mask = np.zeros_like(img)
        right_mask = np.zeros_like(img)
        cv2.drawContours(left_mask, [box], -1, 255, -1)
        cv2.drawContours(right_mask, [box], -1, 255, -1)
        left_mask[:, mid:] = 0
        right_mask[:, :mid] = 0

        return (left_mask, right_mask)
    if len(page_like) >= 2:
        page_like.sort(key=lambda p: cv2.minAreaRect(p[0])[0][0])  # sort by x center
        left_page, right_page = page_like[:2]
        return (left_page, right_page)
    return []

def main(img_name):
    output_folder = create_output_folder()
    # high_blur(img_name, f"{output_folder}/images/")
    # processed_images = preprocess_book_image(img_name, f"{output_folder}/images/")
    img = load_image(img_name)
    img_name = "01_original.png"
    cv2.imwrite(os.path.join(output_folder, img_name), img)

    img = preprocess(img)
    img_name = "02_preprocessed.png"
    cv2.imwrite(os.path.join(output_folder, img_name), img)

    img = morph_cleanup(img)
    img_name = "03_morph_cleanup.png"
    cv2.imwrite(os.path.join(output_folder, img_name), img)

    page_like = find_bright_blobs(img)
    print(f"Found {len(page_like)} page-like regions")

    pages = find_pages(img, page_like)
    print(f"Found {len(pages)} pages")

    for i, page in enumerate(pages):
        page_img = np.zeros_like(img)
        cv2.drawContours(page_img, [page], -1, 255, -1)
        page_name = f"page_{i}.png"
        cv2.imwrite(os.path.join(output_folder, page_name), page_img)


if __name__ == "__main__":
    img_name = sys.argv[1]
    main(img_name)
