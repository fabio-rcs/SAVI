#!/usr/bin/env python3

import pickle
from matplotlib import pyplot as plt

def main():
        # ----------------------------------
        # Initialization
        # ----------------------------------

        # Creating graph
        plt.figure()
        plt.xlim(-10, 10)
        plt.ylim(-5, 5)
        plt.grid()
        plt.xlabel('x')
        plt.ylabel('y')

        print('Create a figure')

        # plt.waitforbuttonpress()
        
        # file = open('pts.pk1', 'rb')
        
        # ----------------------------------
        # Execution
        # ----------------------------------

        # Points dict
        pts = {'xs': [], 'ys': []}

        # Save mouse clicks in pts
        while True:

                plt.plot(pts['xs'], pts['ys'],'b.', linewidth=2, markersize=12)
                pt = plt.ginput(1)
                
                if not pt:
                        print('Exiting...')
                        break

                print('pt = ' + str(pt))

                pts['xs'].append(pt[0][0])
                pts['ys'].append(pt[0][1])

                print(pts)
        
        # ------------------------------------------
        # Termination
        # ------------------------------------------

        file = open('pts.pkl', 'wb')
        pickle.dump(pts, file)        
        file.close()

if __name__ == '__main__':
        main()