#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import open3d as o3d
from copy import deepcopy
import numpy as np

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
    point_cloud_original = o3d.io.read_point_cloud('../data//Factory/factory_isolated.ply')
    
    # ------------------------------------------
    # Execution
    # ------------------------------------------    
    point_cloud = deepcopy(point_cloud_original) 

    #KDTree organizes points in coordinates and makes the neighbor finding process a lot faster
    #Through finding the neighbors, it can create a plane and after, create a normal to that plane
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    #Changes orientation of each normal, so that they point up (positive in z direction)
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [point_cloud]

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    entities.append(frame)


    #Draws entities
    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'],
                                    point_show_normal=True)

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    if cv2.waitKey(0) == ord('q'):
        exit()

if __name__ == "__main__":
    main()
