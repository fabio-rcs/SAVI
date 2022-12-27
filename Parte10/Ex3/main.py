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
			"boundingbox_max" : [ 0.57555947434725718, 0.5559764837161838, 0.5 ],
			"boundingbox_min" : [ -0.51337807678615865, -0.54930516761886095, -0.29956547945857637 ],
			"field_of_view" : 60.0,
			"front" : [ 0.70791938390023512, 0.54720887791513151, 0.44655636803115295 ],
			"lookat" : [ -0.26660283967507298, 0.0034889759092748022, 0.5775805475939817 ],
			"up" : [ -0.40352431826536206, -0.2055497884179546, 0.89158140909835548 ],
			"zoom" : 1.7599999999999998
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
    
    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [p.pcd]

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    # Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

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
