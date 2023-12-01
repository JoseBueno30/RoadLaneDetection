import numpy as np
import cv2
import time
import matplotlib
import matplotlib.pyplot as plt
import threading
import Lane
import LaneFinder
import PerspectiveTransformation
import PreProcessing

if __name__ == '__main__':
    UNWARPED_SIZE = (1280, 720)
    WARPED_SIZE = (600, 500)
    video = cv2.VideoCapture('video/challenge_video.mp4')
    image = cv2.imread('video/frame134.jpg', -1)
    PerspectiveTransform = PerspectiveTransformation.PerspectiveTransform()

    if not video.isOpened():
        print("Error opening video")


    ret, initial_frame = video.read()
    cv2.imshow('tFrame', initial_frame)
    initial_frame = PerspectiveTransform.forward(initial_frame)
    tframe = PreProcessing.binarize(initial_frame)

    left_starting_point, right_starting_point = LaneFinder.getStartingPoints(initial_frame)

    i=0
    left_line = None
    right_line = None
    road = None

    left_curvature = None;
    right_curvature = None;

    sanity_counter_left = 0;
    sanity_counter_right = 0;

    left_line = Lane.Lane(left_starting_point)
    right_line = Lane.Lane(right_starting_point)

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            tframe = np.copy(frame)
            if i == 0:
                start = time.time()
                tframe = PerspectiveTransform.forward(tframe)

                left_line_thread = threading.Thread(target=left_line.find, args=(tframe,))
                right_line_thread = threading.Thread(target=right_line.find, args=(tframe,))

                left_line_thread.start()
                right_line_thread.start()

                left_line_thread.join()
                right_line_thread.join()

                #road = np.hstack(([left_line.getFit()], [np.flipud(right_line.getFit())]))

                i=0;
            else:
                i+=1;


            lane_img = np.zeros_like(frame)

            lane_img = cv2.polylines(lane_img, [left_line.getFit()], False, (255, 0, 255), 3)
            lane_img = cv2.polylines(lane_img, [right_line.getFit()], False, (255, 0, 255), 3)

            # # Draw the lane onto the warped blank image
            #cv2.fillPoly(lane_img, np.int_([road]), (200, 0, 0))
            lane_img = PerspectiveTransform.backward(lane_img)

            frame = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)

            cv2.imshow('tFrame', frame)

            print(time.time()-start)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

