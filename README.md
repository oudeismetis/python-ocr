# Python OCR Experiment

## Quick Commands

```
python -m slow_smart_ocr test_data/arms_length/camera_image.jpg --visualize
python -m stereo_merge test_data/stereo/
python -m cpp_prep test_data/test_images/hp_434.jpeg --visualize
python -m best_case_ocr test_data/test_images/redwall_70.jpg
```

### OpenCV and OCR
```
python -m smart_ocr test_data/test_images/hp_434.jpeg
python -m book_as_qr_code test_data/test_images/hp_434.jpeg
```

## Install
```
brew install tesseract
```

## References
https://towardsdatascience.com/create-simple-optical-character-recognition-ocr-with-python-6d90adb82bb8
https://medium.com/analytics-vidhya/using-tesseract-with-python-1cadbe37e756

## CURRENT NOTES

`claude_exp.py` is pretty great (!!)
It seems to solve the header and footer problem.
Largely it's doing what I'm already doing, but downscaled (speed) with a couple of 
simple steps to bridge the large gaps between text blocks.

It does NOT solve the `new_chapter` problem.

`claude_exp_3.py` is showing similar promise and kind of does OK with `new_chapter`.

Could maybe go with one of those and resolve everything else with some possible rules:
1. If the 2 largest blocks are on top of each other, merge them as one.
1. When controlled for skew, the 2 largest blocks should be ~ the same width.
1. The 2 largest blocks should be no further from each other than X% of page text width. 
1. Extend the `Y` of both blocks to match each other. So same height.
1. The `X` of the 2 blocks can not overlap each other at all.
1. The start `X` of the left block can't be too close to image edge. Same for the end `X` of the right block. If true, ignore that block and don't OCR.

BUT....
This is not object tracking. So maybe table this for the moment and take a hard run at real time object tracking of a book?
Could also maybe quickly implement some of this in C++ to see if we can solve most of our current bugs.
