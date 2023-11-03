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
    video = cv2.VideoCapture('video/project_video.mp4')
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
    left_line = None
    right_line = None

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            tframe = np.copy(frame)


            if i == 0:

                tframe = PreProcessing.histogram_equalization(tframe)
                tframe = PerspectiveTransform.forward(tframe)
                tframe, out = PreProcessing.binarize(tframe)

                left_starting_point, right_starting_point = LaneFinder.getStartingPoints(tframe)

                midpoint = int(tframe.shape[1] / 2)
                left_line, tframe = LaneFinder.findLane(tframe, left_starting_point)
                right_line, tframe = LaneFinder.findLane(tframe, right_starting_point)
                tframe = PerspectiveTransform.backward(tframe)

                frame = cv2.addWeighted(frame, 1, tframe, 0.5, 0)

                #ploty = np.linspace(0, tframe.shape[0] - 1, tframe.shape[0])

                #pts_left = np.array([np.transpose(np.vstack([left_line, ploty]))])
                #pts_left = np.hstack((pts_left))

                #pts_right = np.array([np.transpose(np.vstack([right_line, ploty]))])
                #pts_right = np.hstack((pts_right))

                #frame = PerspectiveTransform.forward(gr)



                i=0;
            else:
                i+=1;


            lane_img = np.zeros_like(frame)


            lane_img = cv2.polylines(lane_img, np.int_([left_line]), False, (0, 255, 0), 3)
            lane_img = cv2.polylines(lane_img, np.int_([right_line]), False, (0, 255, 0), 3)
            lane_img = PerspectiveTransform.backward(lane_img)

            #frame = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)

            cv2.imshow('tFrame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

