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

def extract_elbow_angles(mmpose_json):
    """Extract elbow angles from MMPose JSON format."""
    with open(mmpose_json, 'r') as f:
        data = json.load(f)
   # print(data['meta_info'].keys())
   # print(data['meta_info']["keypoint_id2name"])
   # print(data['instance_info'][-1]["frame_id"]) ## from 1
   # print(data['instance_info'][-1]["instances"][0]["keypoints"]) ## always check if only one BB
    elbow_angles = {}
    # 'right_shoulder', '7'
    # 'right_elbow', '9'
    # 'right_wrist', '11'
    # 'right_hip', '13'
    for frame in data['instance_info']:  # Assuming 'annotations' contains frames
        keypoints = np.array(frame["instances"][0]["keypoints"])
        
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
        
        elbow_angles[frame['frame_id']-1] = {
            'angle_elbow_2d': angle_elbow_2d,
            'angle_chip_2d': angle_chip_2d,
            'angle_elbow_3d': angle_elbow_3d,
            'angle_chip_3d': angle_chip_3d
        }
    
    return elbow_angles

def overlay_angles_on_video(angles, video_path, output_path):
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
            elbow_2d = dict_angles["angle_elbow_2d"]
            elbow_3d = dict_angles["angle_elbow_3d"]
            angle_chip_2d = dict_angles["angle_chip_2d"]
            angle_chip_3d = dict_angles["angle_chip_3d"]
            cv2.putText(frame, f"Angle elbow: 3d->{elbow_3d:.2f}deg 2d->{elbow_2d:.2f}deg", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Angle chip-arm: 3d->{angle_chip_3d:.2f}deg 2d->{angle_chip_2d:.2f}deg", (50, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 0), 2)
        
        out.write(frame)
        frame_id += 1
    
    cap.release()
    out.release()

def get_metrics(file_dir, filename):
    name = filename
    input_json = f"{file_dir}inference_results/3d/results_{name}.json"  # Replace with actual path
    input_video = f"{file_dir}inference_results/3d/{name}.mp4"
    output_video = f"{file_dir}inference_results/3d/{name}_angles.mp4"
    angles = extract_elbow_angles(input_json)
    overlay_angles_on_video(angles, input_video, output_video)
