#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import open3d as o3d
from copy import deepcopy
import numpy as np
from point_cloud_processing import PointCloudProcessing

def main():

    #Initial view parameters
    view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 2.1920225246207465, 2.5861017692224975, 0.82104994623420746 ],
			"boundingbox_min" : [ -2.271826466235042, -2.5269300182961949, -0.58877501755379946 ],
			"field_of_view" : 60.0,
			"front" : [ 0.92068753041036999, -0.037583802458774768, 0.38848671681229202 ],
			"lookat" : [ -0.1708748615795507, 0.76664288193145436, 1.7203222409784105 ],
			"up" : [ -0.39030026556571834, -0.087593200116837633, 0.91651139326940256 ],
			"zoom" : 0.66120000000000023
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    p = PointCloudProcessing() #Calls the class 
    p.loadPointCloud('../data/scene.ply') #Gives the filename to class
    
    # ------------------------------------------
    # Execution
    # ------------------------------------------    
    p.preProcess() #PreProcessing

    p.transform(-110, 0, 0, 0, 0, 0) #Rotate in x
    p.transform(0, 0, 52, 0, 0, 0) #Rotate in z
    p.transform(0, 0, 0, 1.1, -0.8, 0.37) #Translation in xyz

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [p.pcd]

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)


    #Draws entities
    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])
    # ------------------------------------------
    # Termination
    # ------------------------------------------
    if cv2.waitKey(0) == ord('q'):
        exit()

if __name__ == "__main__":
    main()
