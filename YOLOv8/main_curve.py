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
    rolling_avg = 0
    index = 0
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
        cutoff_points = [] 
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
                    
                    if z_distance > 0.05:
                        #print("---------------Distance: ", z_distance)  


                        # Using your data
                        sorted_points = np_points[sorted_indices]
                        rounded_z_values = np.around(z_values, 2)
                        unique_z = np.unique(rounded_z_values)

                        mean_xy_points = []
                        for z in unique_z:
                            z_level_points = sorted_points[rounded_z_values[sorted_indices] == z]
                            mean_xy = np.mean(z_level_points[:, :2], axis=0)
                            mean_xy_points.append(mean_xy)
                        mean_xy_points = np.array(mean_xy_points)

                        # 1. Select the upper part of the asparagus. For this example, let's take the top 70%.
                        # You can adjust this as per your data.
                        upper_fraction = 0.5
                        n_upper_points = int(upper_fraction * len(unique_z))
                        upper_points = mean_xy_points[:n_upper_points]
                        upper_z_values = unique_z[:n_upper_points]

                        # 2. Fit a polynomial. Degree can be adjusted based on the shape.
                        degree = 2
                        p_x = np.polyfit(upper_z_values, upper_points[:, 0], degree)
                        p_y = np.polyfit(upper_z_values, upper_points[:, 1], degree)

                        # 3. Extrapolate to get the base point.
                        z_base = unique_z[-1]
                        x_base = np.polyval(p_x, z_base)
                        y_base = np.polyval(p_y, z_base)

                        base_point = np.array([x_base, y_base, z_base])

                        # 1. Generate the points on the fitted curve.
                        z_curve = np.linspace(min(unique_z), max(unique_z), len(unique_z))
                        x_curve = np.polyval(p_x, z_curve)
                        y_curve = np.polyval(p_y, z_curve)

                        curve_points = np.vstack((x_curve, y_curve, z_curve)).T

                        # 2. Create a PointCloud for the curve (different color for distinction).
                        curve_pcd = o3d.geometry.PointCloud()
                        curve_pcd.points = o3d.utility.Vector3dVector(curve_points)
                        curve_pcd.paint_uniform_color([0, 0, 0])  # RGB green color for the curve

                        # Create a PointCloud object for the mean_xy_points
                       # pcd = o3d.geometry.PointCloud()
                        #pcd.points = o3d.utility.Vector3dVector(mean_xy_points)

                        # Create another PointCloud for the base_point (with a different color for distinction)
                        #base_pcd = o3d.geometry.PointCloud()
                        #base_pcd.points = o3d.utility.Vector3dVector([base_point])
                        #base_pcd.paint_uniform_color([1, 0, 0])  # RGB red color for base point

                        # 3. Visualize everything.
                        coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        o3d.visualization.draw_geometries([downsampled_pcd, coord_axis, curve_pcd], window_name='Asparagus Spine Visualization with Curve')

                        """
                        sorted_points = np_points[sorted_indices]
                        rounded_z_values = np.around(z_values, 2)
                        unique_z = np.unique(rounded_z_values)

                        # 1. Compute mean X and Y position for each unique Z value
                        mean_xy_points = []
                        for z in unique_z:
                            z_level_points = sorted_points[rounded_z_values[sorted_indices] == z]
                            mean_xy = np.mean(z_level_points[:, :2], axis=0)
                            mean_xy_points.append(mean_xy)
                        mean_xy_points = np.array(mean_xy_points)

                        # 2. Calculate Euclidean distance between each consecutive mean point
                        distances = np.linalg.norm(np.diff(mean_xy_points[::-1], axis=0), axis=1)




                        
                        # 3. Identify deviations
                        window_size = 5
                        variances = [np.var(distances[i:i+window_size]) for i in range(len(distances)-window_size)]
                        deviation_index = np.argmax(variances)

                        # Check if deviation_index is zero
                        if deviation_index == 0:
                            print("No significant deviation found. Using all median points to approximate curve.")
                            curve_points = mean_xy_points[::-1]  # Use all median points in reverse order
                            z_values_curve = unique_z[::-1]
                        else:
                            # Use points from the top (lowest Z) to the deviation to predict the curve
                            curve_points = mean_xy_points[::-1][:deviation_index+1]
                            z_values_curve = unique_z[::-1][:deviation_index+1]

                        # 4. Predict the curve - Fit a polynomial separately for X and Y values with respect to Z
                        coeff_x = np.polyfit(z_values_curve, curve_points[:, 0], deg=2)
                        coeff_y = np.polyfit(z_values_curve, curve_points[:, 1], deg=2)
                        polynomial_x = np.poly1d(coeff_x)
                        polynomial_y = np.poly1d(coeff_y)

                        # Generate points on the predicted curve, extending up to Z value of 0.59
                        z_values_curve_extended = np.linspace(unique_z[0], 0.59, len(unique_z))
                        curve_points_predicted_x = polynomial_x(z_values_curve_extended)
                        curve_points_predicted_y = polynomial_y(z_values_curve_extended)
                        curve_points_predicted = np.vstack((curve_points_predicted_x, curve_points_predicted_y)).T


                        # Visualization

                        # Add Z values to the predicted curve points
                        curve_points_3d = np.column_stack((curve_points_predicted, z_values_curve_extended))

                        # Convert the 3D curve points into a PointCloud object
                        curve_pcd = o3d.geometry.PointCloud()
                        curve_pcd.points = o3d.utility.Vector3dVector(curve_points_3d)

                        # Highlight the base point using a distinct color
                        colors = np.array([[0, 1, 0] for _ in range(len(curve_pcd.points))])  # Green for curve points
                        colors[-1] = [1, 0, 0]  # Red for the base point
                        curve_pcd.colors = o3d.utility.Vector3dVector(colors)

                        # Combine the original point cloud with the curve points for visualization
                        combined_pcd = downsampled_pcd + curve_pcd

                        # Determine the Z value corresponding to the deviation_index
                        cutoff_z = unique_z[::-1][deviation_index]

                        filtered_points = np_points[np_points[:, 2] <= cutoff_z]
                        
                        # Create a new point cloud with the filtered points
                        filtered_pcd = o3d.geometry.PointCloud()
                        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
                        

                        # Compute the mean X and Y position for this Z value
                        cutoff_point_xy = mean_xy_points[::-1][deviation_index]
                        cutoff_point = np.array([cutoff_point_xy[0], cutoff_point_xy[1], cutoff_z])
                        cutoff = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(cutoff_point)

                        median_spheres = []
                        # Create spheres for each median point
                        for idx, z in enumerate(unique_z):
                            median_point_xy = mean_xy_points[idx]
                            median_point = np.array([median_point_xy[0], median_point_xy[1], z])
                            
                            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005).translate(median_point)
                            median_spheres.append(sphere)

                        coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                        o3d.visualization.draw_geometries([coord_axis, downsampled_pcd,*median_spheres], window_name="Los Pointos Cloudos")
                        o3d.visualization.draw_geometries([coord_axis, curve_pcd], window_name="Los Pointos Cloudos with Curve")
                        """


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
                        #o3d.visualization.draw_geometries([coord_axis, downsampled_pcd, top_sphere,bot_sphere], window_name="Los Pointos Cloudos")
                        
                        #o3d.visualization.draw_geometries([coord_axis, downsampled_pcd], window_name="Los Pointos Cloudos")
                        
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