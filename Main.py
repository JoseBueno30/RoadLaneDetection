import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt

import LaneFinder
import PerspectiveTransformation
import PreProcessing

if __name__ == '__main__':
    UNWARPED_SIZE = (1280, 720)
    WARPED_SIZE = (600, 500)
    video = cv2.VideoCapture('video/challenge_video.mp4')
    image = cv2.imread('video/frame134.jpg', -1)
    #image = Thresholding.histogram_equalization(image)
    #image = Thresholding.test(image)
    PerspectiveTransform = PerspectiveTransformation.PerspectiveTransform()

    image = PerspectiveTransform.forward(image)
    image  = cv2.Canny(image,10,65)
    cv2.imshow("image", image)
    if not video.isOpened():
        print("Error opening video")


    i=0

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            tframe = np.copy(frame)

            if i == 0:

                tframe = PreProcessing.histogram_equalization(tframe)
                tframe = PerspectiveTransform.forward(tframe)
                tframe, out = PreProcessing.binarize(tframe)

                #print(LaneFinder.getStartingPoints(tframe))
                leftx, rightx = LaneFinder.getStartingPoints(tframe)
                cv2.line(out, (rightx, 720), (rightx, 0),(0, 0, 255), 5)
                cv2.line(out, (leftx, 0), (leftx, 720), (0, 255, 0), 5)
                tframe = PerspectiveTransform.backward(out)


                i=0;
            else:
                i+=1;

            cv2.imshow('tFrame', tframe)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

