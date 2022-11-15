#!/usr/bin/env python3

import pickle
from matplotlib import pyplot as plt
from models import Line
from colorama import Fore, Style
from scipy.optimize import least_squares


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------

    # Load file with points
    file = open('pts.pkl', 'rb') # rb = read binary
    pts = pickle.load(file)
    file.close()

    
    # Create figure
    plt.ion() # Interactive mode on
    plt.figure('Iterations')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title('Scipy Line')
    print('Created graph.')

    # Draw ground truth pts
    plt.plot(pts['xs'], pts['ys'], '.k', linewidth=2, markersize=12)


    # Define the model
    line = Line(pts) 
    best_line = Line(pts) 
    best_error = 1E6 # a very large number to start

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    
    # Set new values
    line.randomizeParams()

    result = least_squares(line.objectiveFunction, [line.m, line.b], verbose=2)

    # ------------------------------------------
    # Termination
    # ------------------------------------------


if __name__ == "__main__":
    main()
