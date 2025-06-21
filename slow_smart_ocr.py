import logging
import random
import traceback
import os
from datetime import datetime

import argparse
from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def bbox_debug_images(img, filtered_regions, clusters, debug_folder):
    """
    Debug visualization for morphological clustering results
    
    Args:
        img: Original image
        filtered_regions: List of MSER regions after preprocessing
        clusters: List of cluster lists from morphological clustering
        debug_folder: Output folder for debug images
    """
    if not debug_folder:
        return
    
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), 
              (255, 0, 255), (0, 255, 255)]  # Green, Red, Blue, Yellow, Magenta, Cyan
    
    # 1. Show all filtered regions (before clustering)
    all_regions_vis = img.copy()
    for region in filtered_regions:
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(all_regions_vis, [hull], True, (0, 255, 0), 1)
    
    logger.info(f"Writing image to: {debug_folder}/03_all_filtered_regions.jpg")
    cv2.imwrite(f"{debug_folder}/03_all_filtered_regions.jpg", all_regions_vis)
    
    # 2. Show morphological mask (intermediate step)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for region in filtered_regions:
        cv2.fillPoly(mask, [region], 255)
    
    # Apply morphological closing
    kernel_size = min(img.shape[:2]) // 50
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    logger.info(f"Writing image to: {debug_folder}/04_morphological_mask.jpg")
    cv2.imwrite(f"{debug_folder}/04_morphological_mask.jpg", closed)
    
    # 3. Show clustered regions with different colors
    cluster_vis = img.copy()
    
    # Create a mapping from region to cluster
    region_to_cluster = {}
    for cluster_id, cluster_regions in enumerate(clusters):
        for region in cluster_regions:
            # Use region as key (convert to tuple for hashing)
            region_key = tuple(region.flatten())
            region_to_cluster[region_key] = cluster_id
    
    # Draw regions colored by cluster
    unclustered_count = 0
    for region in filtered_regions:
        region_key = tuple(region.flatten())
        if region_key in region_to_cluster:
            cluster_id = region_to_cluster[region_key]
            color = colors[cluster_id % len(colors)]
        else:
            color = (128, 128, 128)  # Gray for unclustered (noise)
            unclustered_count += 1
        
        hull = cv2.convexHull(region.reshape(-1, 1, 2))
        cv2.polylines(cluster_vis, [hull], True, color, 1)
    
    logger.info(f"Writing image to: {debug_folder}/05_clustered_regions.jpg")
    cv2.imwrite(f"{debug_folder}/05_clustered_regions.jpg", cluster_vis)
    
    # 4. Show final bounding boxes for main clusters (with outlier removal)
    bbox_vis = img.copy()
    
    def get_convex_hull_bbox(cluster_regions):
        """
        Use convex hull of centroids instead of all points
        """
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

    
    final_corners = []
    text_y = 80
    for i, cluster_regions in enumerate(clusters):
        if cluster_regions:  # Make sure cluster is not empty
            # Get tight bounding box with outlier removal
            # bbox_result = get_tight_bounding_box(cluster_regions, outlier_percentile=10)
            bbox_result = get_convex_hull_bbox(cluster_regions)
            # final_corners.append(np.array(bbox_result, dtype=np.float32))
            final_corners.append(bbox_result)
            if bbox_result is not None:
                regions_used = 999
                hull_points = bbox_result.reshape((-1, 1, 2)).astype(np.int32)

                # Draw bounding box
                color = colors[i % len(colors)]
                cv2.drawContours(bbox_vis, [hull_points], -1, color, 2)
                
                # Add cluster label
                cv2.putText(bbox_vis, f'Page {i + 1}', (50, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 2.5, color, 4)
                text_y += 80
                
                logger.info(f"Cluster {i + 1}: {regions_used}/{len(cluster_regions)} regions used")
    
    logger.info(f"Writing image to: {debug_folder}/06_final_bounding_boxes.jpg")
    cv2.imwrite(f"{debug_folder}/06_final_bounding_boxes.jpg", bbox_vis)
    
    # logger.info summary stats
    logger.info(f"\n=== Clustering Summary ===")
    logger.info(f"Total filtered regions: {len(filtered_regions)}")
    logger.info(f"Number of clusters found: {len(clusters)}")
    logger.info(f"Unclustered regions: {unclustered_count}")
    logger.info(f"Kernel size used: {kernel_size}")
    for i, cluster in enumerate(clusters):
        logger.info(f"Cluster {i + 1}: {len(cluster)} regions")
    logger.info("=========================\n")
    return final_corners

def cluster_regions_morphological(regions, img_shape):
    # Create binary mask from all regions
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for region in regions:
        cv2.fillPoly(mask, [region], 255)
    
    # Morphological closing to connect nearby regions
    # Adjust kernel size based on expected text spacing
    kernel_size = min(img_shape[:2]) // 50  # Try 100?
    # kernel_size = max(kernel_size, 10)  # But not too small
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
        
        if len(cluster_regions) > 10:  # Filter small clusters (noise)
            clusters.append(cluster_regions)
    
    # Sort clusters by size and return the 2 largest
    clusters.sort(key=len, reverse=True)
    return clusters[:2] if len(clusters) >= 2 else clusters

def detect_text_regions_mser(img, debug_folder=None):
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
    logger.info(f"Found {len(regions)} regions using MSER")
    sample_size = max(1, len(regions) // 8)
    filtered_regions = random.sample(regions, sample_size)
    logger.info(f"Filtered down to {len(filtered_regions)} regions")

    clusters = cluster_regions_morphological(filtered_regions, img.shape)
    logger.info("Done with Clustering")
    
    if debug_folder:
        mser_vis = img.copy()
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.polylines(mser_vis, [hull], True, (0, 255, 0), 1)
        logger.info(f"Writing image to: {debug_folder}/02_mser_regions.jpg")
        cv2.imwrite(f"{debug_folder}/02_mser_regions.jpg", mser_vis)
    return bbox_debug_images(img, filtered_regions, clusters, debug_folder)

def crop_image(image, final_corners, debug_folder, idx=0):
    # Get the bounding rectangle of the convex hull
    x, y, w, h = cv2.boundingRect(final_corners.astype(np.int32))
    padding = 20
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
    hull_adjusted = final_corners - [x, y]
    hull_points = hull_adjusted.reshape((-1, 1, 2)).astype(np.int32)

    # Fill the convex hull area in the mask
    cv2.fillPoly(mask, [hull_points], 255)

    # Apply the mask to get only the hull area
    cropped_hull = cv2.bitwise_and(cropped_rect, cropped_rect, mask=mask)
    """
    logger.info(f"Writing image to: 07_{idx}_cropped.jpg")
    cv2.imwrite(f"{debug_folder}/07_{idx}_cropped.jpg", cropped_hull)
    return cropped_hull


def main(img_name, visualize=False):
    """
    Main function to process the book image with corner detection and OCR.
    """
    now = datetime.now()
    try:
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_folder = f"ocr_debug_output_{timestamp}"
        images_folder = f"{debug_folder}/images"

        os.makedirs(images_folder, exist_ok=True)
        
        # Load the image using OpenCV to avoid color issues
        img = cv2.imread(img_name)
        if img is None:
            logger.info(f"Error: Could not load image {img_name}")
            return None
        
        logger.info(f"Processing image: {img_name}")
        logger.info(f"Image shape: {img.shape}")
        
        # Save original image for reference
        cv2.imwrite(f"{images_folder}/01_original.jpg", img)
        
        # Detect book corners
        logger.info("Detecting book corners...")
        corners = detect_text_regions_mser(img, debug_folder)
        # logger.info(f"Detected corners: {corners}")

        for idx, corner in enumerate(corners):
            cropped_image = crop_image(img, corner, debug_folder, idx)
            # Perform OCR on the corrected image
            # logger.info("Performing OCR...")
            # corrected_pil = Image.fromarray(corrected_img)
            # text = image_to_string(corrected_pil)
            
            # Save OCR result to file
            # with open(f"{debug_folder}/ocr_result.txt", "w", encoding="utf-8") as f:
            #     f.write(text)
        logger.info(f"All steps completed in {datetime.now() - now}")
    except Exception as e:
        logger.info(f"Error processing image: {e}")
        logger.info(f"Stack trace: {traceback.format_exc()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from book images with corner detection")
    parser.add_argument("image", help="Path to the input image file")
    parser.add_argument("--visualize", "-v", action="store_true", 
                       help="Save visualization of detected corners")
    
    args = parser.parse_args()
    
    main(args.image, args.visualize)
