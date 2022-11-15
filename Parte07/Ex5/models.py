#!/usr/bin/env python3

import math
from random import uniform
import matplotlib.pyplot as plt
import numpy as np


class Sinusoidal():
    """Defines the model of a sinusoidal function
    """

    def __init__(self, gt):

        self.gt = gt
        self.randomizeParams()
        self.first_draw = True
        self.xs_for_plot = list(np.linspace(-10, 10, num=500))


    def randomizeParams(self):
        self.a = uniform(-10, 10)
        self.b = uniform(-10, 10)
        self.c = uniform(-10, 10)
        self.d = uniform(-10, 10)

    def getY(self, x): # y = mx + b
        return self.a*math.sin(self.b*(x-self.c)) + self.d

    def getYs(self, xs):
        """Retrieves a list of ys by applying the model to a list of xs
        """
        ys = []
        for x in xs:
            ys.append(self.getY(x))
        return ys
            
    def objectiveFunction(self, params):
        self.a = params[0]
        self.b = params[1]
        self.c = params[2]
        self.d = params[3]

        residuals = []

        for gt_x, gt_y in zip(self.gt['xs'], self.gt['ys']):
            y = self.getY(gt_x) # compute y using the current model parameters
            residual = abs(y - gt_y)
            residuals.append(residual)
            
        # error is the sum of the residuals
        error = sum(residuals)

        # Draw for visualization
        self.draw('b--')
        plt.waitforbuttonpress(0.01)
        return error

    def draw(self, color):
       
        if self.first_draw:
            self.draw_handle = plt.plot(self.xs_for_plot, self.getYs(self.xs_for_plot), color, linewidth=2)
            self.first_draw = False
        else:
            plt.setp(self.draw_handle, data=(self.xs_for_plot, self.getYs(self.xs_for_plot)))  # update lm