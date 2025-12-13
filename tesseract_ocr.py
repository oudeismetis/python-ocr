import sys

from PIL import Image
from pytesseract import image_to_string
import numpy as np
import cv2

def clean_up_noise(img_name):
    # normalization, thresholding and image blur.
    img_orig = Image.open(img_name)
    img = np.array(img_orig)
    norm_img = np.zeros((img.shape[0], img.shape[1]))
    img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    img = cv2.GaussianBlur(img, (1, 1), 0)
    return image_to_string(img)

def main(img_name):
    text = image_to_string(Image.open(img_name))
    if not text:
        print('Simple version failed!!!!')
        text = clean_up_noise(img_name)
    print(text)


if __name__ == "__main__":
    # https://towardsdatascience.com/create-simple-optical-character-recognition-ocr-with-python-6d90adb82bb8
    # img_name = 'test.png'
    # img_name = 'test_noisy.png'
    img_name = sys.argv[1]
    main(img_name)
