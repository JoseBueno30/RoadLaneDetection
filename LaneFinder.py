import numpy as np
import cv2
import math

def getStartingPoints(line_mask):
    line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)
    graph = np.sum(line_mask[2*(line_mask.shape[0]//3): , :], axis=0)
    midpoint = int(graph.shape[0]/2)

    left_starting_point = 400 + np.argmax(graph[400:midpoint])
    right_starting_point = midpoint + np.argmax(graph[midpoint:900])

    return left_starting_point, right_starting_point

def findLane(lane_image, starting_point):
    nwindows = 20
    window_height = int(lane_image.shape[0]/nwindows)
    window_width = 20
    minpixel = 40

    nonzero = lane_image.nonzero()
    nonzerox = np.array(nonzero[1])
    nonzeroy = np.array(nonzero[0])

    window_pos = starting_point

    lane_pixels = []

    for window in range (nwindows):
        win_top = lane_image.shape[0] - (window*window_height)
        win_bottom = lane_image.shape[0] - ((window+1)*window_height)
        win_left = window_pos - (window_width/2)
        win_right = window_pos + (window_width/2)

        window_pixels = ((nonzeroy >= win_bottom) & (nonzeroy < win_top) & (nonzerox >= win_left) & (nonzerox <= win_right)).nonzero()[0]

        cv2.rectangle(lane_image, (int(win_left), int(win_bottom)), (int(win_right), int(win_top)), (0, 255, 0), 2)

        lane_pixels.append(window_pixels)

        if len(window_pixels) > minpixel:
            window_pos = int(np.mean(nonzerox[window_pixels]))

    lane_pixels = np.concatenate(lane_pixels)

    lane_pixels_x = nonzerox[lane_pixels]
    lane_pixels_y = nonzeroy[lane_pixels]

    lane_fit = None

    if len(lane_pixels_x) != 0:
        lane_fit = np.polyfit(lane_pixels_y, lane_pixels_x, 2)

    print(lane_fit)


    lane_image[nonzeroy[lane_pixels], nonzerox[lane_pixels]] = [255, 0, 0]

    lane_image[nonzeroy[lane_pixels], nonzerox[lane_pixels]] = [255, 0, 0]

    ploty = np.linspace(0, lane_image.shape[0] - 1, lane_image.shape[0])
    lane_fit = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
    pts_left = np.array([np.transpose(np.vstack([lane_fit, ploty]))])
    pts_left = np.hstack((pts_left))

    frame = cv2.polylines(lane_image, np.int_([pts_left]), False, (0, 255, 255), 5)

    return frame