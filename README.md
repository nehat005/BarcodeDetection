# BarcodeDetection

This project comprises Python implementation of two methodologies to perform the task of barcode detection.

## Overview
- Adopt code from [PyImageSearch](https://pyimagesearch.com/2014/11/24/detecting-barcodes-images-python-opencv/)
- Present basic overview of algorithm along with basic concepts in this repository for understanding purpose.
- Implement a more robust Object Detection algorithm using HOG + Linear SVM

## Algorithm

The general outline of the algorithm is to:

1. Compute the Scharr gradient magnitude representations in both the x and y direction.
2. Subtract the y-gradient from the x-gradient to reveal the barcoded region.
3. Blur and threshold the image.
4. Apply a closing kernel to the thresholded image.
5. Perform a series of dilations and erosions.
6. Find the largest contour in the image, which is now presumably the barcode

