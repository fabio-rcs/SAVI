#!/usr/bin/env python3

from random import uniform
import matplotlib.pyplot as plt


class Line():
    """Defines the model of a line segment
    """

    def __init__(self, gt):

        self.gt = gt
        self.randomizeParams()
        self.first_draw = True


    def randomizeParams(self):
        self.m = uniform(-2, 2)
        self.b = uniform(-5, 5)

    def getY(self, x): # y = mx + b
        return self.m * x + self.b

    def objectiveFunction(self):
        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x) # compute y using the current model parameters
            residual = abs(y - gt_y)
            residuals.append(residual)
            
        # error is the sum of the residuals
        error = sum(residuals)
        return error

    def draw(self, color):
        xi = -10
        xf = 10
        yi = self.getY(xi)
        yf = self.getY(xf)

        if self.first_draw:
            self.draw_handle = plt.plot([xi, xf], [yi, yf], color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handle, data=([xi, xf], [yi, yf]))  # update lm