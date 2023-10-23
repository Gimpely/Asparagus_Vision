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
#----------------------------------------------------------------
    begin_time = time.time()
    start_time = time.time()
#----------------------------------------------------------------
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

    results = model.track(source=frame, conf=0.6, iou=0.5,  boxes=False, tracker="bytetrack.yaml",  persist=True, show_labels=False, show_conf=False)
    #results = model.track(frame, persist=True)
    result = results[0]
#----------------------------------------------------------------
    print("Detection time: ", (time.time() - begin_time)*1000)
    begin_time = time.time()
#----------------------------------------------------------------
    annotated_frame = results[0].plot()

    # Display the annotated frame
    #cv2.imshow("YOLOv8 Tracking", annotated_frame)

    object_masks = []
    if result.masks is not None:
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        ###################################
        all_points = []
        depth_data = np.asanyarray(aligned_depth_frame.get_data())
        all_y_coords, all_x_coords = np.where(depth_data > 0) 
        for y, x in zip(all_y_coords, all_x_coords):
            depth = aligned_depth_frame.get_distance(x, y)
            if depth > 0:
                point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                all_points.append(point)
                
        all_pcd = o3d.geometry.PointCloud()
        if all_points:
            all_pcd.points = o3d.utility.Vector3dVector(all_points)
            all_pcd, ind = all_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        ##################################

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
        bottom_mean_points = []
        bottom_points = [] 
        for mask in object_masks:
            pcd = o3d.geometry.PointCloud()
            points = []
            y_coords, x_coords = np.where(mask)

            for y, x in zip(y_coords, x_coords):
                depth = aligned_depth_frame.get_distance(x, y)
                if depth > 0:
                    point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], depth)
                    points.append(point)
            if points:
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                downsampled_pcd = pcd#pcd.voxel_down_sample(voxel_size=0.01)

                angle_radians = np.radians(45)
                flip_transform = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                                        [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                                        [0, 0, 0, 1]])
                downsampled_pcd.transform(flip_transform)

                points = np.asarray(downsampled_pcd.points)
                if len(downsampled_pcd.points) > 500:
                        

                    object_point_clouds.append(downsampled_pcd)
                    

                    # Getting the top and bottom points
                    NofP = 20
                    np_points = np.asarray(downsampled_pcd.points)
                    z_values = np_points[:, 2]

                    sorted_indices = np.argsort(z_values)  # Sort points by z-coordinate in descending order
                    top_indices = sorted_indices[:NofP]  # Get the indices of the top N points
                    top_mean_point = np.mean(np_points[top_indices], axis=0)  # Compute the mean of the top N points
                    top_mean_points.append(top_mean_point)

                    bottom_indices = sorted_indices[-NofP:]
                    bottom_mean_point = np.mean(np_points[bottom_indices], axis=0)
                    bottom_mean_points.append(bottom_mean_point)

                    z_distance = abs(top_mean_point[2] - bottom_mean_point[2])
                    
                    if z_distance > 0.1:
                        print("---------------Distance: ", z_distance)  

                        sorted_points = np_points[sorted_indices]
                        rounded_z_values = np.around(z_values, 3)  # rounding to group by z-level
                        unique_z = np.unique(rounded_z_values)
                        variances_x = []
                        variances_y = []
                        for z in unique_z:
                            level_points = sorted_points[rounded_z_values[sorted_indices] == z]
                            variance_x = np.var(level_points[:, 0])  # variance in x
                            variance_y = np.var(level_points[:, 1])  # variance in y
                            variances_x.append(variance_x)
                            variances_y.append(variance_y)

                        # Step 3: Detect spike in variance
                        #threshold = np.median(variances)  # Example threshold, can be adjusted
                        #spike_indices = [i for i, v in enumerate(variances) if v > threshold]
                                    
                        


                        #if not (np.all(np.isnan(top_mean_point)) or np.all(np.isnan(bottom_mean_point))):  

                            #delta = bottom_mean_point[2] - top_mean_point[2]
                            #if delta > 0.15:

                        inverse_flip_transform = np.linalg.inv(flip_transform)
                        all_mean_points = top_mean_points + bottom_mean_points
                        transformed_all_points = []

                        for point in all_mean_points:
                            # Homogeneous coordinates: append 1 for the transformation
                            homogeneous_point = np.append(point, 1)
                            
                            # Apply the transformation
                            transformed_point = np.dot(inverse_flip_transform, homogeneous_point)
                            
                            # Drop the homogeneous coordinate and keep only x, y, z
                            transformed_point = transformed_point[:3]
                            transformed_all_points.append(transformed_point)

                        # Split the transformed points back into top and bottom
                        transformed_top_points = transformed_all_points[:len(top_mean_points)]
                        transformed_bottom_points = transformed_all_points[len(top_mean_points):]

                        #print(f"bottom_mean_point: {bottom_mean_point} and the top_mean_point: {top_mean_point} with delta: {delta}")
                        coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        top_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(top_mean_point)
                        bot_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(bottom_mean_point)
                        tspheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.007).translate(point) for point in transformed_top_points]
                        bspheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.007).translate(point) for point in transformed_bottom_points]
                        print(f" --------- Num of points in pcd: {len(downsampled_pcd.points)}")
                        print(f" --------- Top point {top_mean_point}")
                        o3d.visualization.draw_geometries([coord_axis, downsampled_pcd, top_sphere,bot_sphere], window_name="Los Pointos Cloudos")
                        plt.figure(figsize=(10, 6))
                        plt.plot(variances_x, unique_z, '-o', label='Variance in X', color='blue')

                        # Plot variance for Y
                        plt.plot(variances_y, unique_z, '-o', label='Variance in Y', color='green')

                        plt.axvline(x=0.05, color='r', linestyle='--', label='Threshold')  # Example threshold line
                        plt.ylabel('Z Value')
                        plt.xlabel('Variance')
                        plt.title('Variance in X and Y across Z Levels for Asparaguses')
                        plt.legend()
                        plt.grid(True)
                        plt.gca().invert_yaxis()
                        plt.show()
                        #o3d.visualization.draw_geometries( [all_pcd, coord_axis] + tspheres)# + bspheres)


                

        
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