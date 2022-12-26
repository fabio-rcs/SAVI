#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import open3d as o3d
from matplotlib import cm
from copy import deepcopy

class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud #Define point cloud in this class

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) #Paints the plane in color from colormap

    def segment(self, distance_threshold=0.2, ransac_n=3, num_iterations=500):
        #Here we use the ransac algorithm to find planes, with the parameters defined in line 20
        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text

def main():

    #Initial view parameters
    view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.64659847104728452, -0.73557234430037255, 0.20209835115549113 ],
			"lookat" : [ -2.6904689128479808, -4.8985999303177508, 4.103672835468978 ],
			"up" : [ -0.088160207154392309, 0.19109989889036769, 0.97760350169104138 ],
			"zoom" : 0.26119999999999988
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")
    point_cloud_original = o3d.io.read_point_cloud('../data//Factory/factory_floorless.ply')
    
    number_of_planes = 6
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    point_cloud = deepcopy(point_cloud_original) 
    planes = []
    
    #Run consecutive plane detections
    while True: 

        plane = PlaneDetection(point_cloud) #Create a new plane instance
        point_cloud = plane.segment() #New point cloud are the outliers of this plane detection
        print(plane)

        #Colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3] #Gets only R, G and B from the colormap
        plane.colorizeInliers(r=color[0], g=color[1], b=color[2]) #Colorize plane

        planes.append(plane) 

        if len(planes) >= number_of_planes: #Stop detection planes when they reach the defined number
            break   

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [x.inlier_cloud for x in planes]
    entities.append(point_cloud)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up']) #Draw planes

    # ------------------------------------------
    # Termination
    # ------------------------------------------

    if cv2.waitKey(0) == ord('q'):
            exit()

if __name__ == "__main__":
    main()
