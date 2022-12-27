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
			"boundingbox_max" : [ 0.90000000000000002, 0.90000000000000002, 0.5 ],
			"boundingbox_min" : [ -0.90000000000000002, -0.90000000000000002, -0.29999999999999999 ],
			"field_of_view" : 60.0,
			"front" : [ 0.68332635260182362, 0.56056728726209615, 0.46779206095390602 ],
			"lookat" : [ -0.22863384360365879, -0.0422660778435631, -0.013949648578316556 ],
			"up" : [ -0.41368948771981329, -0.23066818457473523, 0.88071175555633086 ],
			"zoom" : 1.0199999999999991
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

    p.crop(-0.9, -0.9, -0.3, 0.9, 0.9, 0.3) #Creates a bbox of points to keep
    #Order: (x,y,z) min and (x,y,z) max
    
    outliers = p.findPlane() #Find the table plane
    p.inliers.paint_uniform_color([0,1,0]) #Paints the plane in green

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [p.inliers, outliers]

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    #Draws entities and show PointCloud in the defined view
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
