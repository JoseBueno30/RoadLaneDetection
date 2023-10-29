import numpy as np
import cv2
import math


def histogram_equalization(img):
    """ Applies histogram equalization to an image.
            Args:
                img: Input image
            Returns:
                equalized_histogram_img: Equalized histogram image
    """
    equalized_histogram_img = np.copy(img)

    return equalized_histogram_img


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
    grayscale_binary = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_binary = histogram_equalization(grayscale_binary)
    ret, grayscale_binary = cv2.threshold(grayscale_binary, 190, 255, cv2.THRESH_BINARY)

    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    img_lab[:, :, 2] = histogram_equalization(img_lab[:, :, 2])
    #img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    binary_img = np.zeros_like(img_lab)
    binary_img[(img_lab[:, :, 2] < 122) | (grayscale_binary > 1) ] = 255

    img[(binary_img > 25)] = 255
    return img

def houghTransform(image):
    # Convert to RGB and get gray image
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    gray = np.copy(image)

    # Blur the gray image
    gray = gaussian_smoothing(gray, 2, 5)



    # Apply Canny algorithm
    edges = cv2.Canny(gray, 255, 255, apertureSize=3)


    # Search for lines using Hough transform
    rho = 1
    theta = np.pi / 180
    threshold = 150
    lines = cv2.HoughLinesP(edges, rho, theta, threshold,
                            minLineLength=40, maxLineGap=40)
    # For each line
    for line in lines:
        # Draw the line in the RGB image
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show resultant image
    return image