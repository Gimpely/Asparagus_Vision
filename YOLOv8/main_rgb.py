import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs       
import os
import math
import open3d as o3d
import time



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
cfg.enable_device_from_file(r"/home/student/Desktop/Asparagus_Vision-main/YOLOv8/L515_posnetek5.bag")
profile = pipe.start(cfg)

model = YOLO("model3s.pt")


while True:

    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame:
        break  
    
    frame = GetBGR(color_frame)
    color = np.asanyarray(color_frame.get_data())

    colorizer = rs.colorizer()
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    aligned_depth_frame = frameset.get_depth_frame()
    colorized_depth_aligned = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    
    height, width, channels = frame.shape

    results = model.track(source=frame, conf=0.6, iou=0.5,  boxes=False, tracker="bytetrack.yaml",  persist=True, show_labels=False, show_conf=False)
    #results = model.track(frame, persist=True)
    result = results[0]

    annotated_frame = results[0].plot()

    # Display the annotated frame
    
    base_points = []
    points_to_display = []
    object_masks = []
    rolling_avg = 0
    index = 0
    if result.masks is not None:
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        for seg in result.masks.xyn:
            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)
            # Fill the mask with the segment
            cv2.fillPoly(mask, [segment], 255)
            
            object_masks.append(mask)
            
        combined_segmented = np.zeros_like(colorized_depth_aligned)

        segmented_objects = []

        
        object_point_clouds = []
        top_mean_points = []
        
        cutoff_points = [] 

        for mask in object_masks:
            pcd = o3d.geometry.PointCloud()
            points = []
            y_coords, x_coords = np.where(mask)

            bottom_y = max(y_coords)

            # corresponding_x = x_coords[np.where(y_coords == max_y)]
            # base_point = (int(np.mean(corresponding_x)), max_y)
            # base_points.append(base_point)

            for i in range(5):
                current_y = bottom_y - i*5  # Moving upwards every 10 pixels
                corresponding_x = x_coords[np.where(y_coords == current_y)]
                # Check if there are any x-values for this y-coordinate
                if len(corresponding_x) > 0:
                    median_x = int(np.median(corresponding_x))
                    points_to_display.append((median_x, current_y))

            

    for point in points_to_display:
        cv2.circle(frame, point, radius=2, color=(0, 0, 255), thickness=-1)  # Green color, filled circle
        cv2.imshow("YOLOv8 Tracking", frame)


            # for y, x in zip(y_coords, x_coords):
            #     depth = aligned_depth_frame.get_distance(x, y)
            #     if depth > 0:
            #         point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
            #         points.append(point)

            # if points:
            #     pcd.points = o3d.utility.Vector3dVector(points)
            #     pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
            #     downsampled_pcd = pcd#pcd.voxel_down_sample(voxel_size=0.01)

            #     angle_radians = np.radians(45)
            #     flip_transform = np.array([[1, 0, 0, 0],
            #                             [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
            #                             [0, np.sin(angle_radians), np.cos(angle_radians), 0],
            #                             [0, 0, 0, 1]])
            #     downsampled_pcd.transform(flip_transform)

            #     points = np.asarray(downsampled_pcd.points)
            #     if len(downsampled_pcd.points) > 500:
                        

            #         object_point_clouds.append(downsampled_pcd)
                    

            #         # Getting the top and bottom points
            #         NofP = 20
            #         np_points = np.asarray(downsampled_pcd.points)
            #         z_values = np_points[:, 2]

            #         sorted_indices = np.argsort(z_values)  # Sort points by z-coordinate in descending order
            #         top_indices = sorted_indices[:NofP]  # Get the indices of the top N points
            #         top_mean_point = np.mean(np_points[top_indices], axis=0)  # Compute the mean of the top N points
            #         top_mean_points.append(top_mean_point)

            #         bottom_indices = sorted_indices[-NofP:]
            #         bottom_mean_point = np.mean(np_points[bottom_indices], axis=0)
            #         bottom_mean_points.append(bottom_mean_point)


            #         z_distance = abs(top_mean_point[2] - bottom_mean_point[2])
                    
            #         if z_distance > 0.05:
            #             print("----------------------------------------------------------")
            #             sorted_points = np_points[sorted_indices]
            #             rounded_z_values = np.around(z_values, 3)
            #             unique_z = np.unique(rounded_z_values)

            

        
            #TODO
            #-Omeji base tocko da je lahko le do dolocene visine, nesme bit nad tlemi
            #-Bottom point naj bi bil na zunanji strani / najbljizji tocki do izhodisca

            
    else:
        combined_segmented = np.zeros_like(colorized_depth_aligned)



    # Visualize the results on the frame
    
    
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()