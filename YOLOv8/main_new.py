import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs       
import os
import math
import open3d as o3d
import time
import matplotlib.pyplot as plt


def GetBGR(frame_color):
    # Input: Intel handle to 16-bit YU/YV data
    # Output: BGR8
    H = frame_color.get_height()
    W = frame_color.get_width()
    Y = np.frombuffer(frame_color.get_data(), dtype=np.uint8)[0::2].reshape(H,W)
    UV = np.frombuffer(frame_color.get_data(), dtype=np.uint8)[1::2].reshape(H,W)
    YUV =  np.zeros((H,W,2), 'uint8')
    YUV[:,:,0] = Y
    YUV[:,:,1] = UV
    BGR = cv2.cvtColor(YUV,cv2.COLOR_YUV2BGR_YUYV)
    return BGR


# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_device_from_file(r"C:\Users\J\Desktop\Asparagus_deteciton\YOLO\YOLOv8\L515_posnetek5.bag")
profile = pipe.start(cfg)

model = YOLO("model3s.pt")

#vis = o3d.visualization.Visualizer()
#vis.create_window()

begin_time = 0
start_time = 0
while True:

    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame:
        break  # No more frames, exit the loop
    
    frame = GetBGR(color_frame)
    color = np.asanyarray(color_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth_aligned = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    
    height, width, channels = frame.shape

    results = model.track(source=frame, conf=0.6, iou=0.5, boxes=True, tracker="bytetrack.yaml", persist=True, show_labels=False, show_conf=False)
    result = results[0]

    annotated_frame = results[0].plot()

    # Extract bounding box data
    bounding_boxes = result.boxes.xyxy
    depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    for bbox in bounding_boxes:
        x1, y1, x2, y2 = map(int, bbox)  # Convert to integers

        # Extract the region of interest from the depth frame
        
        roi = depth_image[y1:y2, x1:x2]

        # Convert the ROI to a point cloud
        pcd = o3d.geometry.PointCloud()
        points = []
        y_coords, x_coords = np.where(roi > 0)  # Only consider valid depth values
        for y, x in zip(y_coords, x_coords):
            global_y = y + y1
            global_x = x + x1
            depth = aligned_depth_frame.get_distance(global_x, global_y)
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [global_x, global_y], depth)
                points.append(point)
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        #Donwsample the point cloud
        downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.001)
        downsampled_pcd, ind = downsampled_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

        #Rotate the point cloud
        angle_radians = np.radians(45)
        flip_transform = np.array([[1, 0, 0, 0],
                                [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                                [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                                [0, 0, 0, 1]])
        pcd.transform(flip_transform)

        # Segment the ground plane
        plane_model, inliers = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
        asparagus_pcd = pcd.select_by_index(inliers, invert=True)
        coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        points = np.asarray(asparagus_pcd.points)
        #z_threshold = np.percentile(points[:, 2], 5)  # Adjust percentile as needed
        filtered_points = points[points[:, 2] < 0.59]

        base_point_idx = np.argmax(filtered_points[:, 2])  # Find index of the point with the highest Z-value
        base_point = filtered_points[base_point_idx]

        # Convert filtered_points back to a PointCloud object for visualization
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # Create a sphere at the base_point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
        sphere.paint_uniform_color([0, 0, 0])  # Color the sphere red for visibility
        sphere.translate(base_point)

        # Visualize the point cloud with the sphere

        o3d.visualization.draw_geometries([downsampled_pcd, coord_axis], window_name='Asparagus raw')
        o3d.visualization.draw_geometries([asparagus_pcd, coord_axis], window_name='Asparagus without plane')
        o3d.visualization.draw_geometries([filtered_pcd, sphere, coord_axis])

        

            #TODO
            #-Omeji base tocko da je lahko le do dolocene visine, nesme bit nad tlemi
            #-Bottom point naj bi bil na zunanji strani / najbljizji tocki do izhodisca



           
                
            


    


    # Visualize the results on the frame
    
