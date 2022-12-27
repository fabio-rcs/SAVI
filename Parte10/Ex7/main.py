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
from matplotlib import cm
from more_itertools import locate
import copy
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering


#Draws the ICP result for the cereal box fitting
def draw_registration_result(source, target, transformation): 
    view = {
                "class_name" : "ViewTrajectory",
                "interval" : 29,
                "is_loop" : False,
                "trajectory" : 
                [
                    {
                        "boundingbox_max" : [ 0.60237595358218832, 0.5908036245657472, -0.11457819573799555 ],
                        "boundingbox_min" : [ 0.44443189957596407, 0.39540735461712756, -0.42293220931545733 ],
                        "field_of_view" : 60.0,
                        "front" : [ -0.97176958211300257, 0.17542961722291259, 0.15776035205634778 ],
                        "lookat" : [ -0.099476831337040675, 0.49967640643177713, -0.083297523662816189 ],
                        "up" : [ 0.22033628535239694, 0.43572957732029705, 0.87269218903653678 ],
                        "zoom" : 2.0
                    }
                ],
                "version_major" : 1,
                "version_minor" : 0
            }
    #We use temporary variables for this
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp],
                                        zoom=view['trajectory'][0]['zoom'],
                                        front=view['trajectory'][0]['front'],
                                        lookat=view['trajectory'][0]['lookat'],
                                        up=view['trajectory'][0]['up'])

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
                        "boundingbox_min" : [ -0.90000000000000002, -0.90000000000000002, -0.10000000000000003 ],
                        "field_of_view" : 60.0,
                        "front" : [ 0.57428340315845261, 0.72415504342386261, 0.38183510307530683 ],
                        "lookat" : [ -0.23064077816298553, -0.2045093977126011, 0.17408966530741635 ],
                        "up" : [ -0.36479928025136604, -0.19118507801017759, 0.91124626258455921 ],
                        "zoom" : 0.79999999999999893
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
    p.preProcess(voxel_size=0.01) #PreProcessing

    #?Center the origin in the table
    p.transform(-110, 0, 0, 0, 0, 0) #Rotate in x
    p.transform(0, 0, 52, 0, 0, 0) #Rotate in z
    p.transform(0, 0, 0, 1.1, -0.8, 0.37) #Translation in xyz

    #?Creates a bbox of points to keep
    p.crop(-0.9, -0.9, -0.3, 0.9, 0.9, 0.3)
    #Order: (x,y,z) min and (x,y,z) max
    
    outliers = p.findPlane() #Find the table plane
    p.inliers.paint_uniform_color([0,1,0]) #Paints the plane in green

    #?Now we isolate the objects (cluster the points for each one)
    cluster_idxs = list(outliers.cluster_dbscan(eps=0.035, min_points=70, print_progress=True))
    #Cluster returns points associated to an ID and gives the same ID for points in the defined range
    #If eps small, no object is going to be detected
    
    object_idxs = list(set(cluster_idxs))
    object_idxs.remove(-1) #Removes -1 cluster ID (-1 are the points not clustered)

    number_of_objects = len(object_idxs)
    
    #Color each cluster with a different color using colormap
    colormap = cm.Pastel1(list(range(0,number_of_objects)))

    #Create the objects list
    objects = []

    #Here we find the points for each object and reunite them 
    for object_idx in object_idxs:

        #Locate the points, comparing each cluster id to the object id, grouping them and converts the result to list because it's easier
        object_point_idxs = list(locate(cluster_idxs, lambda x: x == object_idx))
        object_points = outliers.select_by_index(object_point_idxs)
        
        #Create a dictionary to define the objects
        d = {}
        d['idx'] = str(object_idx)
        d['points'] = object_points 
        d['color'] = colormap[object_idx, 0:3] #Define color of the object
        d['points'].paint_uniform_color(d['color']) #Paints the point in the defined color
        d['center'] = d['points'].get_center() #Gets center of Point Cloud
        objects.append(d) #Add the dict of this object to the list

    #?Find the cereal box
    #First we load the model (source)
    cereal_box_model = o3d.io.read_point_cloud('../data/cereal_box_2_2_40.pcd')

    #Then we compare each object (target) to the model
    for object_idx, object in enumerate(objects):
        print("\nApply point-to-point ICP to object " + str(object['idx']) )

        #We define the transformation matrix as the identity matrix (in this case there's no need to move the objects)
        trans_init = np.asarray([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0], 
                                 [0.0, 0.0, 0.0, 1.0]])
        
        #Here we try to fit the source object in the models
        reg_p2p = o3d.pipelines.registration.registration_icp(cereal_box_model, 
                                                              object['points'], 1, trans_init, o3d.pipelines.registration.TransformationEstimationPointToPoint())
        
        print(reg_p2p.inlier_rmse) #Prints fitting error
        
        #Associates error of transformation to the object
        object['rmse'] = reg_p2p.inlier_rmse 
        
        # #Draw the fitting visual result
        # draw_registration_result(cereal_box_model, object['points'], reg_p2p.transformation) 
           
    #To classify the object, we use the smallest fitting value to decide which object is a "cereal box" (the closest to the source model)
    minimum_rmse = 10e8 #Just a very large number to start
    cereal_box_object_idx = None

    for object_idx, object in enumerate(objects):

        if object['rmse'] < minimum_rmse: #Found a new minimum
        
            minimum_rmse = object['rmse']
            cereal_box_object_idx = object_idx

    print('\nThe cereal box is object ' + str(cereal_box_object_idx) + '\n')

    # ------------------------------------------
    # Visualization
    # ------------------------------------------
    
    #Create a list of entities to draw
    entities = []

    #?Create a triangular mesh
    #Iterate for each object
    for object_idx, object in enumerate(objects):
        alpha = 0.02
        print(f"alpha={alpha:.3f}")
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(object['points'], alpha)
        mesh.compute_vertex_normals()
        entities.append(mesh)

    #Create a frame with the orthogonal directions
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.3, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    #Draw bbox
    bbox_to_draw = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(p.bbox)
    entities.append(bbox_to_draw)

    #Draw objects
    for object in objects:
        entities.append(object['points'])


    # #Draws entities and show PointCloud in the defined 3D view
    # o3d.visualization.draw_geometries(entities,
    #                                 zoom=view['trajectory'][0]['zoom'],
    #                                 front=view['trajectory'][0]['front'],
    #                                 lookat=view['trajectory'][0]['lookat'],
    #                                 up=view['trajectory'][0]['up'])

    #?Make a more complex Open3D window to show object labels on top of 3D objects
    app = gui.Application.instance
    app.initialize() #Create an Open3D app

    #App variables
    w = app.create_window("Open3D - 3D Text", 1920, 1080) #Define window name and resolution
    widget3d = gui.SceneWidget()
    widget3d.scene = rendering.Open3DScene(w.renderer)
    widget3d.scene.set_background([0,0,0,1])  #Set black background
    material = rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    material.point_size = 2 * w.scaling

    #Draw entities
    for entity_idx, entity in enumerate(entities):
        widget3d.scene.add_geometry("Entity " + str(entity_idx), entity, material)

    # Draw labels
    for object_idx, object in enumerate(objects):
        label_pos = [object['center'][0], object['center'][1], object['center'][2] + 0.3]

        label_text = 'Object ' + object['idx']
        if object_idx == cereal_box_object_idx:
            label_text += ' (Cereal Box)'

        label = widget3d.add_3d_label(label_pos, label_text)
        label.color = gui.Color(object['color'][0], object['color'][1],object['color'][2])
        label.scale = 2
    
    bbox = widget3d.scene.bounding_box
    widget3d.setup_camera(60.0, bbox, bbox.get_center())
    w.add_child(widget3d)

    app.run()

    # ------------------------------------------
    # Termination
    # ------------------------------------------
    if cv2.waitKey(0) == ord('q'):
        exit()

if __name__ == "__main__":
    main()
