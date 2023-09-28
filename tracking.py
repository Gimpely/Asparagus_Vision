import cv2
from ultralytics import YOLO
import numpy as np
import pyrealsense2 as rs       
import os
import math

def rotMat(theta_degrees=45):
    theta_radians = math.radians(theta_degrees)
    cos_theta = math.cos(theta_radians)
    sin_theta = math.sin(theta_radians)
    return cos_theta, sin_theta


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

    results = model.track(source=frame, conf=0.6, iou=0.5,  boxes=False, tracker="bytetrack.yaml",  persist=True, show_labels=False, show_conf=False)
    #results = model.track(frame, persist=True)
    result = results[0]

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
        for mask in object_masks:
            segmented_object = cv2.bitwise_and(colorized_depth_aligned, colorized_depth_aligned, mask=mask)

            
            combined_segmented = cv2.add(combined_segmented, segmented_object)
    else:
        combined_segmented = np.zeros_like(colorized_depth_aligned)
    cv2.imshow("Segmented Object", combined_segmented)
    cv2.waitKey(1)
    
    # Visualize the results on the frame
    
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLOv8 Tracking", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()