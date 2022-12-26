#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import math
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
    point_cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    
    #Changes orientation of each normal, so that they point up (positive in z direction)
    point_cloud.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))

    #?Find the points that have vertical normal

    angle_tolerance = 10
    vx, vy, vz = 0, 0, 1
    norm_b = math.sqrt(vx**2 + vy**2 + vz**2)
    vertical_idxs = []

    #First we calculate the angle between the normal of the point and the vertical direction, so that we can create a list of points with vertical normal    
    for idx, normal in enumerate(point_cloud.normals):
        nx, ny, nz = normal
        ab = nx*vx + ny*vy + nz*vz
        norm_a = math.sqrt(nx**2 + ny**2 + nz**2)
        angle = math.acos(ab/(norm_a * norm_b)) * 180/ math.pi

    #Second, if the angle is inferior to the tolerance, then we can say it has a vertical normal
        if angle < angle_tolerance: #This point has a "vertical normal"
            vertical_idxs.append(idx)

    #List with vertical normals and list without
    vertical_cloud = point_cloud.select_by_index(vertical_idxs)
    non_vertical_cloud = point_cloud.select_by_index(vertical_idxs, invert=True)

    vertical_cloud.paint_uniform_color([0.5, 0, 1]) #Paints the plane in X color
    
    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [vertical_cloud, non_vertical_cloud]

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
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
