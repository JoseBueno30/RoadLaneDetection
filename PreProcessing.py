import numpy as np
import cv2


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
    img_lab[:, :, 0] = histogram_equalization(img_lab[:, :, 0])
    img_lab[:, :, 2] = histogram_equalization(img_lab[:, :, 2])
    img_hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    img_hls[:, :, 1] = histogram_equalization(img_hls[:, :, 1])

    binary_imge = np.zeros_like(grayscale_binary)
    binary_imge[(img_lab[:, :, 2] < 122) | (grayscale_binary > 1) ] = 255


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 7))
    black = cv2.morphologyEx(img_lab[:, :, 0], cv2.MORPH_TOPHAT, kernel)
    lanes = cv2.morphologyEx(img_hls[:, :, 1], cv2.MORPH_TOPHAT, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    yellow = cv2.morphologyEx(binary_imge, cv2.MORPH_TOPHAT, kernel)

    final_mask = np.zeros_like(img_lab)
    final_mask[(black > 10) | (lanes > 10) | (yellow > 10)] = 255

    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_ERODE, small_kernel, 10)


    img[(final_mask > 25)] = 255
    return final_mask, img
