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
cfg.enable_device_from_file("L515_posnetek5.bag")
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
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    object_masks = []
    if result.masks is not None:
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

        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        object_point_clouds = []
        top_mean_points = []
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

                angle_radians = np.radians(45)
                flip_transform = np.array([[1, 0, 0, 0],
                                        [0, np.cos(angle_radians), -np.sin(angle_radians), 0],
                                        [0, np.sin(angle_radians), np.cos(angle_radians), 0],
                                        [0, 0, 0, 1]])
                pcd.transform(flip_transform)

                object_point_clouds.append(pcd)
                

                
                NofP = 10
                np_points = np.asarray(pcd.points)
                np_points = np.asarray(pcd.points)
                sorted_indices = np.argsort(np_points[:, 2])  # Sort points by z-coordinate in descending order
                top_indices = sorted_indices[:NofP]  # Get the indices of the top N points
                top_mean_point = np.mean(np_points[top_indices], axis=0)  # Compute the mean of the top N points
                top_mean_points.append(top_mean_point)







                cv2.imshow("Segmented mask", mask)
                cv2.waitKey(1)

                segmented_object = cv2.bitwise_and(colorized_depth_aligned, colorized_depth_aligned, mask=mask)
                cv2.imshow("Segmented Object", segmented_object)
                cv2.waitKey(1)

                #for pcd in object_point_clouds:
                #o3d.visualization.draw_geometries([pcd])

                # samo za vizualization
                combined_segmented = cv2.add(combined_segmented, segmented_object)
                
                cv2.imshow("Segmented Object", combined_segmented)
                cv2.waitKey(1)
                coord_axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
                spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.002).translate(point) for point in top_mean_points]
                o3d.visualization.draw_geometries(object_point_clouds + [coord_axis] + spheres)
        
            #TODO
            #Kako simplify sparge
            #Zaznaj top in bottom tocke spargla. Za bottom glej std po visini v horizontali. ce odstopa velik (so tla) jih porezi
            #Transformacija v k.s. robota



           
                
            


            

            
    else:
        combined_segmented = np.zeros_like(colorized_depth_aligned)



    # Visualize the results on the frame
    
    
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()