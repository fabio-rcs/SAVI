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
    ply_point_cloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud('../data/Factory/factory.ply')
    print(pcd)
    print(np.asarray(pcd.points))
    o3d.visualization.draw_geometries([pcd],
                                    zoom=view["trajectory"][0]['zoom'],
                                    # zoom=0.3412,
                                    front=view["trajectory"][0]['front'],
                                    # front=[0.4257, -0.2125, -0.8795],
                                    lookat=view["trajectory"][0]['lookat'],
                                    # lookat=[2.6172, 2.0475, 1.532],
                                    up=view["trajectory"][0]['up'])
                                    # up=[-0.0694, -0.9768, 0.2024])




    # ------------------------------------------
    # Execution
    # ------------------------------------------

    # ------------------------------------------
    # Termination
    # ------------------------------------------


    if cv2.waitKey(0) == ord('q'):
            exit()

if __name__ == "__main__":
    main()
