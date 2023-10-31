import numpy as np
import cv2
import math

def getStartingPoints(line_mask):
    line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)
    graph = np.sum(line_mask[2*(line_mask.shape[0]//3): , :], axis=0)
    midpoint = int(graph.shape[0]/2)
    print("-----------------------------")
    print((graph.shape))
    #print(np.max(graph[:midpoint]))
    #print(graph[np.argmax(graph[:midpoint])])
    print("-----------------------------")
    left_starting_point = 400 + np.argmax(graph[400:midpoint])
    right_starting_point = midpoint + np.argmax(graph[midpoint:900])

    return left_starting_point, right_starting_point