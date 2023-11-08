import numpy as np
import cv2
import math
import PreProcessing

def getStartingPoints(line_mask):
    line_mask = cv2.cvtColor(line_mask, cv2.COLOR_RGB2GRAY)
    graph = np.sum(line_mask[3*(line_mask.shape[0]//4): , :], axis=0)
    midpoint = int(graph.shape[0]/2)

    # left_starting_point = 500 + np.argmax(graph[500:midpoint-70])
    left_starting_point = 450 + int(np.mean(line_mask[650:, 450:midpoint].nonzero()[1]))
    # right_starting_point = midpoint + np.argmax(graph[midpoint:900])
    right_starting_point = midpoint + int(np.mean(line_mask[650:, midpoint:900].nonzero()[1]))

    return left_starting_point, right_starting_point

def findLane(lane_image, starting_point):
    nwindows = 10
    window_height = int(lane_image.shape[0]/nwindows)
    window_width = 120
    minpixel = 750


    window_pos = starting_point

    lane_pixels_x, lane_pixels_y = [],[]

    for window in range (nwindows):
        win_bottom = lane_image.shape[0] - (window*window_height)
        win_top = lane_image.shape[0] - ((window+1)*window_height)
        win_left = window_pos - (window_width/2)
        win_right = window_pos + (window_width/2)

        window = lane_image[int(win_top):int(win_bottom), int(win_left):int(win_right), :]
        window = PreProcessing.histogram_equalization(window)

        processed_image = PreProcessing.binarize(window)

        nonzero = processed_image.nonzero()
        nonzerox = np.array(nonzero[1]) + int(win_left)
        nonzeroy = np.array(nonzero[0]) + int(win_top)

        #lane_image[int(win_top):int(win_bottom), int(win_left):int(win_right), :] = processed_image
        #cv2.rectangle(lane_image, (int(win_left), int(win_bottom)), (int(win_right), int(win_top)), (0, 255, 0), 1)

        lane_pixels_x.append(nonzerox)
        lane_pixels_y.append(nonzeroy)

        if len(nonzerox) > minpixel:

            window_pos = int(np.mean(nonzerox))



    lane_pixels_x = np.concatenate(lane_pixels_x)
    lane_pixels_y = np.concatenate(lane_pixels_y)

    lane_fit = None

    if len(lane_pixels_x) != 0:
        lane_fit = np.polyfit(lane_pixels_y, lane_pixels_x, 2)

    #lane_image[lane_pixels_y, lane_pixels_x] = [255, 0, 0]

    #lane_image[lane_pixels_y, lane_pixels_x] = [255, 0, 0]


    ploty = np.linspace(144, lane_image.shape[0] - 1, lane_image.shape[0])

    lane_fit = lane_fit[0] * ploty ** 2 + lane_fit[1] * ploty + lane_fit[2]
    pts = np.array([np.transpose(np.vstack([lane_fit, ploty]))])
    pts = np.hstack(np.int_(pts))

    return pts, lane_image