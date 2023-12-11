import numpy as np
import cv2
from numba import jit, cuda





def gaussian_smoothing(image, sigma, w_kernel):
    """ Blur and normalize input image.

        Args:
            image: Input image to be binarized
            sigma: Standard deviation of the Gaussian distribution
            w_kernel: Kernel aperture size

        Returns:
            binarized: Blurred image
    """

    # Define 1D kernel
    s = sigma
    w = w_kernel
    kernel_1D = [np.exp(-z * z / (2 * s * s)) / np.sqrt(2 * np.pi * s * s) for z in range(-w, w + 1)]

    # Apply distributive property of convolution
    kernel_2D = np.outer(kernel_1D, kernel_1D)

    # Blur image
    smoothed_img = cv2.filter2D(image, cv2.CV_8U, kernel_2D)

    # Normalize to [0 254] values
    smoothed_norm = np.array(image.shape)
    smoothed_norm = cv2.normalize(smoothed_img, None, 0, 255, cv2.NORM_MINMAX)

    return smoothed_norm


def binarize(img):
    #SEPARATE FUNCTION
    binary = cv2.convertScaleAbs(img, alpha=1.1, beta=-90)
    binary = cv2.cvtColor(binary, cv2.COLOR_RGB2HSV)
    binary[:, :, 1] = np.zeros_like(binary[:, :, 1])
    binary = cv2.cvtColor(binary, cv2.COLOR_HSV2RGB)

    ret, binary = cv2.threshold(binary[:,:,0], int(cv2.mean(binary[:,:,0])[0])+45, 255, cv2.THRESH_BINARY)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    binary_img = np.zeros_like(binary)
    binary_img[(img_lab[:, :, 2] < 115) | (binary > 1) ] = 255


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    black = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)
    lanes = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    yellow = cv2.morphologyEx(binary_img, cv2.MORPH_TOPHAT, kernel)

    #lanes = np.zeros_like(lanes)
    #black = np.zeros_like(black)
    #yellow = np.zeros_like(yellow)
    final_mask = np.zeros_like(binary)
    final_mask[(yellow > 1) | (lanes > 2) | (black > 3)] = 255

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 7))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, small_kernel, 10)
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, small_kernel, 10)
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, small_kernel, 10)


    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 31))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE, small_kernel)

    return final_mask

def Sobel (img):
    kernel_h = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]) * 1 / 4
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.filter2D(img,cv2.CV_8U,kernel_h)

    # small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 7))
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, small_kernel, 10)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = binarize(img)


    #img = binarize(img)

    #binary = cv2.convertScaleAbs(img, alpha=1.1, beta=-80)
    # binary = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # binary[:, :, 1] = np.zeros_like(binary[:, :, 1])
    # #binary[:, :, 2] = np.zeros_like(binary[:, :, 2])
    # binary = cv2.cvtColor(binary, cv2.COLOR_HSV2RGB)
    #
    # ret, binary[:, :, 0] = cv2.threshold(binary[:, :, 0], int(cv2.mean(binary[:, :, 0])[0]) + 45, 255, cv2.THRESH_BINARY)


    return img


def binarize_kmeans(image, it):
    """ Binarize an image using k-means.

        Args:
            image: Input image
            it: K-means iteration
    """

    # Set random seed for centroids
    # image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


    cv2.setRNGSeed(124)

    # Flatten image
    flattened_img = image.reshape((-1, 1))
    flattened_img = np.float32(flattened_img)

    # Set epsilon
    epsilon = 0.2

    # Estabish stopping criteria (either `it` iterations or moving less than `epsilon`)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, it, epsilon)

    # Set K parameter (2 for thresholding)
    K = 2

    # Call kmeans using random initial position for centroids
    _, label, center = cv2.kmeans(flattened_img, K, None, criteria, it, cv2.KMEANS_RANDOM_CENTERS)

    # Colour resultant labels
    center = np.uint8(center)  # Get center coordinates as unsigned integers
    # print(center[0], center[1])
    if(center[1] - center[0]) < 20:
        center[0] = 0
        center[1] = 0
    else:
        center[0] = 0
        center[1] = 255

    flattened_img = center[label.flatten()]  # Get the color (center) assigned to each pixel

    # Reshape vector image to original shape
    binarized = flattened_img.reshape((image.shape))

    # binarized = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)

    # Show resultant image
    # final = np.zeros_like(img)
    # final[binarized > 10] = 255

    return binarized


def Canny(img):
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.Canny(img, 100, 200, apertureSize=3)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


    return img

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 5, 25, 25)