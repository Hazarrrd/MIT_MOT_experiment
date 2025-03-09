import json
import numpy as np
import cv2

def calculate_angle(a, b, c):
    """Calculate the angle between three points (shoulder, elbow, wrist)."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle)

def pixel_distance(point1, point2):
    """Calculate Euclidean distance between two points in pixels."""
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def real_world_distance(point1, point2, scale):
    """Convert pixel distance to real-world distance using a given scale."""
    pixel_dist = pixel_distance(point1, point2)
    return pixel_dist / scale  # scale = pixels per real-world unit

def get_dict_angles(keypoints):
    # COCO format: [x, y, confidence] for each keypoint
    shoulder_right = keypoints[6]  # Right shoulder
    elbow_right = keypoints[8]  # Right elbow
    wrist_right = keypoints[10]  # Right wrist
    chip_right = keypoints[12]  # Right wrist
    #  if frame['frame_id'] == 1:
    #      print(f"{shoulder_right} {elbow_right} {wrist_right} {chip_right}")
    angle_elbow_2d = calculate_angle(shoulder_right, elbow_right, wrist_right)
    angle_chip_2d = calculate_angle(chip_right, shoulder_right, elbow_right)

    out_dict = {
            'angle_elbow_2d': angle_elbow_2d,
            'angle_chip_2d': angle_chip_2d,
        }
    return out_dict

def get_dict_angles_back(keypoints):
    # COCO format: [x, y, confidence] for each keypoint
    shoulder_right = keypoints[6]  # Right shoulder
    elbow_right = keypoints[8]  # Right elbow
    wrist_right = keypoints[10]  # Right wrist
    chip_right = keypoints[12]  # Right wrist
    #  if frame['frame_id'] == 1:
    #      print(f"{shoulder_right} {elbow_right} {wrist_right} {chip_right}")
    angle_elbow_2d = calculate_angle(shoulder_right[[0,2]], elbow_right[[0,2]], wrist_right[[0,2]])
    angle_chip_2d = calculate_angle(chip_right[[0,2]], shoulder_right[[0,2]], elbow_right[[0,2]])

    angle_elbow_3d = calculate_angle(shoulder_right, elbow_right, wrist_right)
    angle_chip_3d = calculate_angle(chip_right, shoulder_right, elbow_right)
    out_dict = {
            'angle_elbow_2d': angle_elbow_2d,
            'angle_chip_2d': angle_chip_2d,
            'angle_elbow_3d': angle_elbow_3d,
            'angle_chip_3d': angle_chip_3d
        }
    return out_dict
        
def extract_elbow_angles(data):
    """Extract elbow angles from MMPose JSON format."""
   # print(data['meta_info'].keys())
   # print(data['meta_info']["keypoint_id2name"])
   # print(data['instance_info'][-1]["frame_id"]) ## from 1
   # print(data['instance_info'][-1]["instances"][0]["keypoints"]) ## always check if only one BB
    elbow_angles = {}
    dists = {}
    # 'right_shoulder', '7'
    # 'right_elbow', '9'
    # 'right_wrist', '11'
    # 'right_hip', '13'
    max_frame = -1
    p4 = [0,0]
    SCALE = pixel_distance([560,470],[633,470]) / 10.5
    for frame in data['instance_info']:
        if len(frame["keypoints_2d"]['__ndarray__']) ==0:
            continue
        #print(frame["keypoints_2d"])
        keypoints = np.array(frame["keypoints_2d"]['__ndarray__'][0])
        if frame['frame_id'] > max_frame:
            max_frame = frame['frame_id']
            p4 = keypoints[10]
            
    for frame in data['instance_info']:  # Assuming 'annotations' contains frames
        if len(frame["keypoints_2d"]['__ndarray__']) ==0:
            continue
        #print(frame["keypoints_2d"])
        keypoints = np.array(frame["keypoints_2d"]['__ndarray__'][0])
        p1 = keypoints[8] ## elbow
        p2 = keypoints[10] ##wrist
        p3 = keypoints[6]  ##arm
      #  SCALE= pixel_distance(p1,p2) / 25
        # COCO format: [x, y, confidence] for each keypoint
        out_dict = get_dict_angles(keypoints)
        dist_dict = {
            "elbow_wrist_dist_pix":pixel_distance(p1,p2),
            "wrist_screen_dist":real_world_distance(p2, p4,SCALE)
        }
        elbow_angles[frame['frame_id']-1] = out_dict
        dists[frame['frame_id']-1] = dist_dict
    
    return elbow_angles, dists

def overlay_angles_on_video(angles,dists, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)), 
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id in angles:
            dict_angles = angles[frame_id]
            
            str_print = f""
            offset = 0
            for key,val in dict_angles.items():
                if not np.isnan(val):
                    try:
                        str_print += (f" || {key}: {int(val)} || ")
                        cv2.putText(frame, f"{key}: {val:.2f}deg", (50+offset, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        offset += 450
                    except ValueError:
                        pass
            
            offset = 0
            cv2.putText(frame, f"elbow_wrist_dist_pix: {dists[frame_id]['elbow_wrist_dist_pix']:.2f}px", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            cv2.putText(frame, f"wrist_screen_dist:: {dists[frame_id]['wrist_screen_dist']:.2f}cm", (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                        
                        
        
        out.write(frame)
        frame_id += 1
    
    cap.release()
    out.release()

def get_metrics(file_dir, filename):
    name = filename
    input_json = f"{file_dir}inference_results/3d/results_{name}.json"  # Replace with actual path
    input_video = f"{file_dir}inference_results/3d/{name}.mp4"
    output_video = f"{file_dir}inference_results/3d/{name}_angles.mp4"
    with open(input_json, 'r') as f:
        data = json.load(f)
    angles,dists = extract_elbow_angles(data)
    overlay_angles_on_video(angles,dists, input_video, output_video)
