import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs       
import os
import math
import open3d as o3d
import time

#version 1
'''
def update_position_with_odometry(tracking_info, time_elapsed):
    # Predict new position based on last known position and time elapsed (using constant speed assumption)
    # This is a simplified 2D odometry prediction, assuming motion in a straight line
    predicted_movement = 0.3 * time_elapsed  # Your assumed speed in m/s
    # Update the current point based on the predicted movement
    tracking_info['current_point'] = (tracking_info['current_point'][0] + predicted_movement,
                                      tracking_info['current_point'][1],
                                      tracking_info['current_point'][2])
    return tracking_info['current_point']
'''
#Version 2
def update_position_with_odometry(tracking_info, time_elapsed, speed=0.3):
    """
    Predict new position based on last known position and time elapsed.
    The camera is facing backward and looking down at a 45-degree angle.
    Movement will increase the Z coordinate and decrease the Y coordinate equally.
    """
    predicted_movement = speed * time_elapsed
    # Calculate the change in the Y and Z coordinates due to the 45-degree camera angle
    movement_yz = predicted_movement / math.sqrt(2)  # Divide by sqrt(2) because of 45-degree angle
    current_point = tracking_info['current_point']
    # Since the camera is facing backwards, increasing Z coordinate means moving forward.
    # The Y coordinate decreases because we're looking down at 45 degrees.
    updated_point = (current_point[0], current_point[1] - movement_yz, current_point[2] + movement_yz)
    return updated_point


def blend_positions(predicted_point, measured_point, odometry_weight=0.2):
    """
    Blend the odometry and measured positions using a weighted average.
    :param predicted_point: The point predicted by odometry.
    :param measured_point: The point measured by the detection system.
    :param odometry_weight: The weight given to the odometry data.
    :return: The blended point.
    """
    # Weights must sum up to 1
    detection_weight = 1 - odometry_weight
    blended_point = tuple(odometry_weight * p + detection_weight * m for p, m in zip(predicted_point, measured_point))
    return blended_point

def calculate_distance(point_a, point_b):
    return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2 + (point_a[2] - point_b[2]) ** 2)

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
#cfg.enable_device_from_file(r"/home/student/Desktop/Asparagus_Vision-main/YOLOv8/L515_posnetek5.bag")
cfg.enable_device_from_file(r"C:\Users\J\Desktop\Asparagus_deteciton\YOLO\YOLOv8\L515_posnetek5.bag")
profile = pipe.start(cfg)

model = YOLO("model3s.pt")

tracked_objects = {}


previous_time = time.time()
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

    points_to_display = []
    object_masks = []

    current_time = time.time()
    time_elapsed = current_time - previous_time  # Time elapsed since last frame was processed
    previous_time = current_time 
    
    predicted_movement_per_detection  = 0.3 * time_elapsed
    #if result.masks is not None:
    if result.masks is not None and results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        threshold_height = 20 

        object_point_clouds = []
        top_mean_points = []
        
        points = []
        points_with_ids = []
        print("track_ids", track_ids)
        for index, object_id in enumerate(track_ids):
       
            seg = result.masks.xyn[index]
            

            # contours
            seg[:, 0] *= width
            seg[:, 1] *= height
            segment = np.array(seg, dtype=np.int32)

            mask = np.zeros((height, width), dtype=np.uint8)

            cv2.fillPoly(mask, [segment], 255)
            
            y_coords, x_coords = np.where(mask)

            bottom_y = max(y_coords)
            if (height - threshold_height) > bottom_y:

                corresponding_x = x_coords[np.where(y_coords == bottom_y)]
                # Check if there are any x-values for this y-coordinate
                if len(corresponding_x) > 0:
                    median_x = int(np.median(corresponding_x))
                    points_to_display.append((median_x, bottom_y))

                    depth = aligned_depth_frame.get_distance(median_x, bottom_y)
                    if depth > 0:
                        point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [median_x, bottom_y], depth)
                        points.append(point)
                        points_with_ids.append((object_id, point))

        if points_with_ids:
            for object_id, point in points_with_ids:
                if object_id not in tracked_objects:
                    tracked_objects[object_id] = {'initial_point': point, 'current_point': point, 'last_seen': current_time}
                else:
                    # Existing object, combine odometry and detection
                    predicted_point = update_position_with_odometry(tracked_objects[object_id], time_elapsed)
                    blended_point = blend_positions(predicted_point, point)
                    tracked_objects[object_id]['current_point'] = blended_point
                    tracked_objects[object_id]['last_seen'] = current_time

        #predict step
        current_tracked_ids = [obj_id for obj_id, _ in points_with_ids]
        for object_id, tracking_info in tracked_objects.items():
            if object_id not in current_tracked_ids:
                time_since_last_seen = current_time - tracking_info['last_seen']
                # Predict new position based on last known position and time elapsed (using constant speed assumption)
                # This is a simplified 2D odometry prediction, assuming motion in a straight line
                predicted_movement = 0.3 * time_since_last_seen  # Your assumed speed in m/s
                # Update the current point based on the predicted movement
                tracking_info['current_point'] = (tracking_info['current_point'][0] + predicted_movement,
                                                  tracking_info['current_point'][1],
                                                  tracking_info['current_point'][2])
                
        # Remove objects that have moved more than 2 meters from their initial detection location
        for object_id in list(tracked_objects.keys()):  # Make a copy of keys to avoid modifying the dict while iterating
            initial_point = tracked_objects[object_id]['initial_point']
            current_point = tracked_objects[object_id]['current_point']
            distance_moved = calculate_distance(initial_point, current_point)
            if distance_moved > 2.0:
                del tracked_objects[object_id]

        print("points",len(points))
    
    for point in points_to_display:
        threshold_height = 20
        line_y = height - threshold_height
        #cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)  
        cv2.circle(frame, point, radius=2, color=(0, 0, 255), thickness=-1)  

    for object_id, tracking_info in tracked_objects.items():
        # Draw the current blended position
        point_3d = tracking_info['current_point']
        pixel_coordinates = rs.rs2_project_point_to_pixel(depth_intrinsics, point_3d)
        cv2.circle(frame, (int(pixel_coordinates[0]), int(pixel_coordinates[1])), radius=4, color=(255, 0, 0), thickness=-1)  # Blue for blended point
        
    cv2.imshow("YOLOv8 Tracking", frame)
    print("points", tracked_objects)

    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()