import logging
import random
import traceback
import os
from datetime import datetime

import argparse
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_convex_hull_bbox(cluster_regions):
    centroids = []
    for region in cluster_regions:
        M = cv2.moments(region)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            centroids.append([cx, cy])
    
    if len(centroids) < 3:
        all_points = np.vstack(cluster_regions)
        return cv2.boundingRect(all_points)
    
    centroids = np.array(centroids, dtype=np.int32)
    hull = cv2.convexHull(centroids)
    return np.squeeze(hull).astype(np.float32)

def cluster_regions_morphological(regions, img_shape):
    # Create binary mask from all regions
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for region in regions:
        cv2.fillPoly(mask, [region], 255)
    
    # Morphological closing to connect nearby regions
    # Adjust kernel size based on expected text spacing
    kernel_size = min(img_shape[:2]) // 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Find connected components
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Group original regions by which connected component they belong to
    clusters = []
    for contour in contours:
        cluster_regions = []
        for region in regions:
            # Check if region centroid is inside this connected component
            M = cv2.moments(region)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                if cv2.pointPolygonTest(contour, (cx, cy), False) >= 0:
                    cluster_regions.append(region)
        
        if len(cluster_regions) > 20:  # Filter small clusters (noise)
            clusters.append(cluster_regions)
    
    # Sort clusters by size and return the 2 largest
    clusters.sort(key=len, reverse=True)
    return clusters[:2] if len(clusters) >= 2 else clusters

def detect_text_regions_mser(img):
    """
    Use MSER to detect text regions and find their bounding corners.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    mser = cv2.MSER_create(
        delta=5,
        min_area=10,
        max_area=300,
        max_variation=0.5,
        min_diversity=0.5,
        max_evolution=200,
        area_threshold=1.01,
        min_margin=0.003,
        edge_blur_size=5
    )
    """
    mser = cv2.MSER_create(
        delta=8,           # Increased from 5 - less sensitive to small variations
        min_area=120,      # Increased from 60 - filter out tiny noise regions
        max_area=7200,     # Decreased from 14400 - avoid huge regions
        max_variation=0.15, # Decreased from 0.25 - more restrictive on variation
        min_diversity=0.3,  # Increased from 0.2 - more diverse regions
        max_evolution=150,  # Decreased from 200 - less evolution steps
        area_threshold=1.02, # Slightly increased from 1.01
        min_margin=0.005,   # Increased from 0.003
        edge_blur_size=3    # Decreased from 5 - less blurring
    )
    """
    
    regions, _ = mser.detectRegions(gray)
    return regions

def crop_image(image, corners, debug_folder, idx=0):
    x, y, w, h = cv2.boundingRect(corners.astype(np.int32))
    padding = 30
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
    h_padded = min(image.shape[0] - y_padded, h + 2 * padding)

    # Crop to the bounding rectangle first
    cropped_hull = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]

    """
    # Create a mask for the convex hull within the cropped area
    mask = np.zeros((h, w), dtype=np.uint8)

    # Adjust hull coordinates to the cropped image coordinate system
    hull_adjusted = corners - [x, y]
    hull_points = hull_adjusted.reshape((-1, 1, 2)).astype(np.int32)

    # Fill the convex hull area in the mask
    cv2.fillPoly(mask, [hull_points], 255)

    # Apply the mask to get only the hull area
    cropped_hull = cv2.bitwise_and(cropped_rect, cropped_rect, mask=mask)
    """
    logger.info(f"Writing image to: 07_{idx}_cropped.jpg")
    cv2.imwrite(f"{debug_folder}/07_{idx}_cropped.jpg", cropped_hull)
    return cropped_hull


def main(img_name):
    now = datetime.now()
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_folder = f"ocr_debug_output_{timestamp}"
        images_folder = f"{debug_folder}/images"
        os.makedirs(images_folder, exist_ok=True)
        
        img = cv2.imread(img_name)
        regions = detect_text_regions_mser(img)
        logger.info(f"Found {len(regions)} regions using MSER")
        sample_size = max(1, len(regions) // 8)
        filtered_regions = random.sample(regions, sample_size)
        logger.info(f"Filtered down to {len(filtered_regions)} regions")

        clusters = cluster_regions_morphological(filtered_regions, img.shape)
        logger.info("Done with Clustering")
        corners = []
        for cluster_regions in clusters:
            if cluster_regions:
                bbox_result = get_convex_hull_bbox(cluster_regions)
                corners.append(bbox_result)

        for idx, corner in enumerate(corners):
            cropped_image = crop_image(img, corner, debug_folder, idx)
        logger.info(f"All steps completed in {datetime.now() - now}")
    except Exception as e:
        logger.info(f"Error processing image: {e}")
        logger.info(f"Stack trace: {traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect book text and crop it out")
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args()
    main(args.image)
