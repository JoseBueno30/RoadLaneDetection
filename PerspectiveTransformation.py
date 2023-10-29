import numpy as np
import cv2

def sharpen_img(img):
    gb = cv2.GaussianBlur(img, (5, 5), 20.0)
    return cv2.addWeighted(img, 2, gb, -1, 0)


def lin_img(img, s=1.0, m=0.0):  # Compute linear image transformation img*s+m
    img2 = cv2.multiply(img, np.array([s]))
    return cv2.add(img2, np.array([m]))


def contr_img(img, s=1.0):  # Change image contrast; s>1 - increase
    m = 127.0 * (1.0 - s)
    return lin_img(img, s, m)

class PerspectiveTransform:
    def __init__(self):
        self.src = np.float32([[300, 450], [400, 680], [880, 680], [980, 450]])
        self.dst = np.float32([[0, 0], [500, 720], [780, 720], [1280, 0]])

        self.TransformationMatrix = cv2.getPerspectiveTransform(self.src, self.dst)
        self.TransformationMatrix_inv = cv2.getPerspectiveTransform(self.dst, self.src)

    def forward (self, img):
        transformed_img = np.copy(img)
        x = img.shape[1]
        y = img.shape[0]
        transformed_img = cv2.warpPerspective(transformed_img, self.TransformationMatrix, (x, y), cv2.INTER_LINEAR)

        transformed_img = sharpen_img(transformed_img)
        #transformed_img = contr_img(transformed_img, 1.1)


        return transformed_img

    def backward (self, img):
        transformed_img = np.copy(img)
        x = img.shape[1]
        y = img.shape[0]
        transformed_img = cv2.warpPerspective(transformed_img, self.TransformationMatrix_inv, (x, y), cv2.INTER_LINEAR)
        return transformed_img