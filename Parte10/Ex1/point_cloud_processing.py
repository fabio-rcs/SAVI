#!/usr/bin/env python3
# --------------------------------------------------
# Fábio Sousa.
# SAVI, December 2022.
# --------------------------------------------------

import open3d as o3d
from copy import deepcopy
import numpy as np

class PointCloudProcessing():
    def __init__(self):
        pass

    def loadPointCloud(self, filename):
        print("Load a point cloud from " + filename)
        self.pcd = o3d.io.read_point_cloud(filename)
        self.original = deepcopy(self.pcd) # make a backup of the original point cloud

    def preProcess(self,voxel_size=0.02):
        # Downsampling using voxel grid filter
        self.pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size) 
        print('Downsampling reduced point cloud from  ' + str(len(self.original.points)) + ' to ' + str(len(self.pcd.points))+  ' points')

        # Estimate normals
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
        # TODO Is this a good orientation???
        self.pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0., 0., 1.]))