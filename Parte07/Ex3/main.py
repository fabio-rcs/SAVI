#!/usr/bin/env python3

import pickle
from matplotlib import pyplot as plt
from models import Sinusoidal
from colorama import Fore, Style

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
    plt.title('Optimized sinusoidal')
    print('Created graph.')

    # Draw ground truth pts
    plt.plot(pts['xs'], pts['ys'], '.k', linewidth=2, markersize=12)


    # Define the model
    sinusoid = Sinusoidal(pts) 
    best_sinusoid = Sinusoidal(pts) 
    best_error = 1E6 # a very large number to start

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    while True: # iterate setting new values for the params and recomputing the error

        # Set new values
        sinusoid.randomizeParams()

        # Compute error
        error = sinusoid.objectiveFunction()
        print(error)

        if error < best_error: # we found a better model!!!
            best_sinusoid.a = sinusoid.a # copy the best found sinusoid params
            best_sinusoid.b = sinusoid.b
            best_sinusoid.c = sinusoid.c
            best_sinusoid.d = sinusoid.d
            best_error = error # update best error
            print(Fore.RED + 'We found a better model!!!' + Style.RESET_ALL)

        # Draw current model
        sinusoid.draw('b--')
        best_sinusoid.draw('r-')


        plt.waitforbuttonpress(0.01)
        if not plt.fignum_exists(1): # a way to do a clean termination
            print('Exiting...')
            plt.ioff()
            break
        


    # ------------------------------------------
    # Termination
    # ------------------------------------------


if __name__ == "__main__":
    main()
