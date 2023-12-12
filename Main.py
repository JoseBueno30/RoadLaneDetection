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


def drawCurvature(frame, left_aperture, right_aperture):
    aperture = None
    if abs(left_aperture) > abs(right_aperture):
        aperture = left_aperture
    else:
        aperture = right_aperture
    print(abs(aperture))

    if abs(aperture) <= 0.00003:
        direction = "Straight Line"
        cv2.putText(frame, direction, org=(10, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1,
                    color=(255, 255, 255), thickness=1)
    elif aperture < 0:
        direction = "Left Curve"
        cv2.putText(frame, direction,
                    org=(10, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(frame, "Curvature = {:.0f} m".format(min(left_line.curvature, right_line.curvature)),
                    org=(10, 70), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
    else:

        direction = "Right Curve"
        cv2.putText(frame, direction,
                    org=(10, 35), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)
        cv2.putText(frame, "Curvature = {:.0f} m".format(min(left_line.curvature, right_line.curvature)),
                    org=(10, 70), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

def drawPosition(frame, pos):

    cv2.putText(frame, "Distance to centre = {:.2f} m".format(pos),
                org=(10, 115), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(255, 255, 255), thickness=1)

def drawDisplay(frame):
    cv2.rectangle(frame, (5, 5), (505, 130), (60, 60, 60), -1)



def drawlane(lane_img):
    cv2.fillPoly(lane_img, np.int_([road]), (200, 0, 0))


if __name__ == '__main__':
    video = cv2.VideoCapture('video/project_video.mp4')
    PerspectiveTransform = PerspectiveTransformation.PerspectiveTransform()

    if not video.isOpened():
        print("Error opening video")


    ret, initial_frame = video.read()
    cv2.imshow('tFrame', initial_frame)
    initial_frame = PerspectiveTransform.forward(initial_frame)
    left_starting_point, right_starting_point = LaneFinder.getStartingPoints(initial_frame)

    i=1

    left_line = Lane.Lane(left_starting_point)
    right_line = Lane.Lane(right_starting_point)
    road = None

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            start = time.time()


            tframe = PerspectiveTransform.forward(frame)

            left_line_thread = threading.Thread(target=left_line.find, args=(tframe,))
            right_line_thread = threading.Thread(target=right_line.find, args=(tframe,))

            left_line_thread.start()
            right_line_thread.start()

            left_line_thread.join()
            right_line_thread.join()

            left_line_fit = left_line.getFit()
            right_line_fit = right_line.getFit()

            road = np.hstack(([left_line_fit], [np.flipud(right_line_fit)]))

            lane_img = np.zeros_like(frame)

            # lane_img = cv2.polylines(lane_img, [left_line.getFit()], False, (0, 155, 255), 10)
            #lane_img = cv2.polylines(lane_img, [right_line.getFit()], False, (0, 155, 255), 10)

            #print(left_line.getFit())

            position = (1280 // 2 - (right_line_fit[0][0] + left_line_fit[0][0]) // 2) * 3.7 / 700





            # Draw the lane onto the warped blank image
            laneThread = threading.Thread(target=drawlane, args=(lane_img,))
            displayThread = threading.Thread(target=drawDisplay, args=(frame,))
            textThread = threading.Thread(target=drawCurvature, args=(frame, left_line.getPolynomial()[0], right_line.getPolynomial()[0],))
            posThread = threading.Thread(target=drawPosition, args=(frame, position,))

            laneThread.start()
            displayThread.start()
            textThread.start()
            posThread.start()

            laneThread.join()
            displayThread.join()
            textThread.join()
            posThread.join()

            tframe = PreProcessing.binarize(tframe)
            #frame = PerspectiveTransform.backward(tframe)
            #frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            lane_img = PerspectiveTransform.backward(lane_img)
            frame = cv2.addWeighted(frame, 1, lane_img, 0.5, 0)



            cv2.imshow('tFrame', frame)

            #print(time.time()-start)


            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    video.release()
    cv2.destroyAllWindows()

