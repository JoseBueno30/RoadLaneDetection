import numpy as np
import cv2

def color_correction(img):
    corrected = cv2.convertScaleAbs(img, alpha=1.1, beta=-90)
    corrected = cv2.cvtColor(corrected, cv2.COLOR_RGB2HSV)
    corrected[:, :, 1] = np.zeros_like(corrected[:, :, 1])
    corrected = cv2.cvtColor(corrected, cv2.COLOR_HSV2RGB)
    return corrected
def binarize(img):
    binary = color_correction(img)
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    ret, binary = cv2.threshold(binary[:,:,0], int(cv2.mean(binary[:,:,0])[0])+45, 255, cv2.THRESH_BINARY)
    binary_img = np.zeros_like(binary)
    binary_img[(img_lab[:, :, 2] > 155) | (binary > 1) ] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    mask1 = cv2.morphologyEx(binary_img, cv2.MORPH_TOPHAT, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    mask2 = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)
    mask3 = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)

    return combine_masks(binary, mask1, mask2, mask3)

def combine_masks(img, mask1, mask2, mask3):
    final_mask = np.zeros_like(img)
    final_mask[(mask1 > 1) | (mask2 > 2) | (mask3 > 3)] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, kernel, iterations=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, kernel, 10)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, kernel)

    return final_mask

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 5, 25, 25)