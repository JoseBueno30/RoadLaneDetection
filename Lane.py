import numpy as np

import LaneFinder
class Lane:
    def __init__(self, starting_point):
        self.starting_point = starting_point
        self.line = None
        self.fit = np.array([None])
        self.curvature = None
        self.sanity_counter = 0

    def find(self, img):
        self.line, self.curvature, self.sanity_counter, self.fit = LaneFinder.findRoadLane(img, self.starting_point, self.line, self.curvature, self.sanity_counter, self.fit)

    def getFit(self):
        return self.line

    def getPolynomial(self):
        return self.fit