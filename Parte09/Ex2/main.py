#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import cv2
import open3d as o3d
import numpy as np

def main():

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

    ply_point_cloud = o3d.data.PLYPointCloud() #Initializes Open3D

    # ------------------------------------------
    # Execution
    # ------------------------------------------
    pcd = o3d.io.read_point_cloud('../data/Factory/factory.ply') #Reads point cloud file

    plane_model, inliers = pcd.segment_plane(distance_threshold=0.35,
                                            ransac_n=3,
                                            num_iterations=1000) #Plane finder

    [a, b, c, d] = plane_model

    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    print(pcd)
    print(np.asarray(pcd.points))

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0]) #Paints the plane in red
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                    zoom=view["trajectory"][0]['zoom'],
                                    front=view["trajectory"][0]['front'],
                                    lookat=view["trajectory"][0]['lookat'],
                                    up=view["trajectory"][0]['up']) #Draws everything and loads the required view

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    print('Saving PointCloud without plane points (without ground)')
    o3d.io.write_point_cloud('../data/Factory/factory_floorless.ply', outlier_cloud, write_ascii=False, compressed=False, print_progress=False)


    if cv2.waitKey(0) == ord('q'):
            exit()

if __name__ == "__main__":
    main()
