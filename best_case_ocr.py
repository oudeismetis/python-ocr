import argparse
import logging
import traceback
import os
from datetime import datetime

import cv2
# from pytesseract import image_to_string
import pytesseract


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

tess_config = (
    "--psm 6 --oem 1 " + 
    "-c tessedit_char_whitelist= abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.?;\\\'\\\"!$%()*"
)

def draw_low_confidence_boxes(image):
    red = (0, 0, 255)
    orange = (0, 165, 255)
    green = (0, 255, 0)
    # Get OCR results with bounding box data
    print(tess_config)
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, lang="eng", config=tess_config)

    img_with_boxes = image.copy()

    red_count = 0
    orange_count = 0
    green_count = 0
    
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        conf = int(data['conf'][i])
        if conf > 0:
            if conf < 50 and data['text'][i].strip() != '':
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), red, 2)
                red_count += 1
            elif conf < 80 and data['text'][i].strip() != '':
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), orange, 2)
                orange_count += 1
            elif conf < 95 and data['text'][i].strip() != '':
                (x, y, w, h) = (data['left'][i], data['top'][i], data['width'][i], data['height'][i])
                cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), green, 2)
                green_count += 1

    logger.info("\n")
    logger.info("===========================")
    logger.info(f"Red count: {red_count}")
    logger.info(f"Orange count: {orange_count}")
    logger.info(f"Green count: {green_count}")
    logger.info("===========================")
    logger.info(f"Total count: {red_count + orange_count + green_count}")
    logger.info("===========================")
    logger.info("\n")

    return img_with_boxes

def rotate_image(image, angle_degrees):
    """
    Rotates an image (numpy array) by the specified angle in degrees.
    
    Args:
        image (np.ndarray): Input image.
        angle_degrees (float): Rotation angle in degrees (positive = counterclockwise).
    
    Returns:
        np.ndarray: Rotated image.
    """
    (h, w) = image.shape[:2]
    center = (w / 2, h / 2)
    
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    
    rotated = cv2.warpAffine(
        image, rotation_matrix, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE  # You can change this as needed
    )
    
    return rotated

def crop_image(image, x, y, w, h):
    padding = 0
    x_padded = max(0, x - padding)
    y_padded = max(0, y - padding)
    w_padded = min(image.shape[1] - x_padded, w + 2 * padding)
    h_padded = min(image.shape[0] - y_padded, h + 2 * padding)

    cropped_hull = image[y_padded:y_padded+h_padded, x_padded:x_padded+w_padded]
    return cropped_hull


def main(img_name):
    """
    Main function to process the book image with corner detection and OCR.
    """
    now = datetime.now()
    try:
        # Create output directory structure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_folder = f"ocr_debug_output_{timestamp}"

        os.makedirs(debug_folder, exist_ok=True)
        
        # Load the image using OpenCV to avoid color issues
        img = cv2.imread(img_name)
        
        logger.info(f"Processing image: {img_name}")
        logger.info(f"Image shape: {img.shape}")
        
        # Save original image for reference
        cv2.imwrite(f"{debug_folder}/01_original.jpg", img)
        
        rotated_image = rotate_image(img, 2)
        cv2.imwrite(f"{debug_folder}/02_rotated.jpg", rotated_image)

        cropped_image = crop_image(rotated_image, 700, 420, 1320, 2170)
        # Used for dewarp vanilla logic
        # cropped_image = crop_image(rotated_image, 700, 420, 1460, 2170)
        cv2.imwrite(f"{debug_folder}/03_cropped.jpg", cropped_image)

        dewarped_img = cv2.imread("test_data/test_images/hp_434_dewarped.png")
        cv2.imwrite(f"{debug_folder}/04_dewarped.jpg", dewarped_img)


        final_image = dewarped_img
        # final_image = cropped_image

        low_confidence_boxes = draw_low_confidence_boxes(final_image)
        cv2.imwrite(f"{debug_folder}/04_low_confidence.jpg", low_confidence_boxes)

        text = pytesseract.image_to_string(final_image, lang="eng", config=tess_config)
        # text = pytesseract.image_to_string(final_image)
        # Save OCR result to file
        with open(f"{debug_folder}/ocr_result.txt", "w", encoding="utf-8") as f:
            f.write(text)
        logger.info(f"All steps completed in {datetime.now() - now}")
        # breakpoint()
    except Exception as e:
        logger.info(f"Error processing image: {e}")
        logger.info(f"Stack trace: {traceback.format_exc()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from book image")
    parser.add_argument("image", help="Path to the input image file")
    args = parser.parse_args()
    
    main(args.image)
    """
    [Video frames] → [Align] → [Normalize lighting] → [Merge (avg or median)] →
        [Grayscale] → [CLAHE] → [Threshold or Adaptive] → [Morphology] → [OCR]

    Total   <50     <80     <95
    115     24      27      64      Logical rotate and crop
    99      16      27      56          + eng lang, PSM 6, OEM 3
    92      11      23      58      Vanilla dewarp after logical rotate and crop
    92      10      23      59          + eng lang, PSM 6, OEM 3

    """
