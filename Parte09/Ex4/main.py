#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import open3d as o3d
from copy import deepcopy
from more_itertools import locate

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
    
    # ------------------------------------------
    # Execution
    # ------------------------------------------    
    point_cloud = deepcopy(point_cloud_original) 
    print('\nBefore downsampling, point cloud has ' + str(len(point_cloud.points)) + ' points')

    #Downsampling using voxel grid filter
    #*This filter returns a centerpoint for each voxel_size square 
    point_cloud_downsampled = point_cloud.voxel_down_sample(voxel_size=0.1) 
    print('After downsampling, point cloud has ' + str(len(point_cloud_downsampled.points)) + ' points\n')
    

    #Clustering 

    #Get cluster indexes
    cluster_idxs = list(point_cloud_downsampled.cluster_dbscan(eps=0.45, min_points=50, print_progress=True)) 

    possible_values = list(set(cluster_idxs))
    possible_values.remove(-1) #Remove clusters with the value -1
    print(possible_values)

    #Here we just want two clusters, one for the building and another for everything else
    #And we know that the building cluster will be the bigger one
    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))

    cloud_building = point_cloud_downsampled.select_by_index(largest_idxs) 
    cloud_others = point_cloud_downsampled.select_by_index(largest_idxs, invert=True)

    cloud_others.paint_uniform_color([0,0,0.5]) #Paints blue

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = [cloud_building, cloud_others]

    #Draws entities
    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    o3d.io.write_point_cloud('../data/Factory/factory_isolated.ply', cloud_building, write_ascii=False, compressed=False, print_progress=False)
    print('\nPoint cloud data saved in data folder')
    if cv2.waitKey(0) == ord('q'):
        exit()

if __name__ == "__main__":
    main()
