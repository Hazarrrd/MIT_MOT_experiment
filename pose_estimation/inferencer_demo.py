import argparse
import os
import pathlib
import subprocess
import time
import numpy as np

from argparse import ArgumentParser
from datetime import timedelta

from mmpose.apis.inferencers import MMPoseInferencer
from tqdm import tqdm
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Devices:", torch.cuda.device_count())
print("Current Device:", torch.cuda.current_device())
print("Device Name:", torch.cuda.get_device_name(0))



def parse_args() -> argparse.Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", "-i", type=pathlib.Path, help="Input video path or folder path.", required=True)
    parser.add_argument("--output", "-o", type=pathlib.Path, help="Output save dir", required=True)
    parser.add_argument("--compress", "-c", action="store_true", help="True if script should compress output video")

    return parser.parse_args()

def get_predictions(
    input_path: pathlib.Path, output_path: pathlib.Path, model_str="rtmpose-l", compress: bool = False, inferencer = None, out_filename_sufix = ""
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    if not inferencer:
       # inferencer = MMPoseInferencer(pose2d=model_str)
       #MMPoseInferencer("rtmw-x_8xb704-270e_cocktail14-256x192") 
       inferencer = MMPoseInferencer("rtmw-x_8xb320-270e_cocktail14-384x288") 
      #  inferencer = MMPoseInferencer(
      #  det_model = "/home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py",
     #   det_weights = "/home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth",
     #   det_cat_ids=[0],
     #   pose2d="/home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw-x_8xb704-270e_cocktail14-256x192.py",
     #   pose2d_weights="/home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw-dw-x-l_simcc-cocktail14_270e-256x192-20231122.pth",
     # )
    time_start = time.time()
    results = []
    to_test = []
    for prediction in tqdm(inferencer(inputs=str(input_path),num_instances=1,kpt_thr=0.5, batch_size=16, vis_out_dir=str(output_path), draw_bbox=True, thickness = 1, radius = 1)):
        for idx,frame_result in enumerate(prediction["predictions"]):
            frame_data = []
            for bb in frame_result:
                bb_data = {
                        "frame_index": idx,
                        "bounding_box": bb.get("bbox", []),
                        "bb_confidence_score": bb.get("bbox_score", []).tolist(),
                        "keypoints": bb.get("keypoints", []),
                        "confidence_scores": bb.get("keypoint_scores", []),
                }
                bb_data2 = {
                    "frame" : str(input_path),
                    "height" : abs(bb.get("bbox", [])[0][1] - bb.get("bbox", [])[0][3]),
                    "width" : abs(bb.get("bbox", [])[0][0] - bb.get("bbox", [])[0][2]),
                    "keypoints" : bb.get("keypoints", []),
                    "confidence_scores" : np.mean(bb.get("keypoint_scores", [])),
                    "bounding_box": bb.get("bbox", [])
                }
                to_test.append(bb_data2)
                frame_data.append(bb_data)
            results.append(frame_data)

    # Save extracted prediction details to a file
    results_file = output_path / f"{input_path.stem}_predictions.json"
    with open(results_file, "w") as f:
        import json
        json.dump(results, f, indent=4)

    time_passed = time.time() - time_start
    time_passed_str = str(timedelta(seconds=time_passed))
    time_passed_str = time_passed_str.split(".")[0].replace(":", "-")
    print("Time: " + time_passed_str)

    output_name = f"result_{out_filename_sufix}_{input_path.stem}_{model_str}_{time_passed_str}{input_path.suffix}"
    current_name = input_path.name

    if compress:
        subprocess.call(
            [
                "ffmpeg",
                "-i",
                str(output_path / current_name),
                "-fs",
                "800M",
                str(output_path / f"{input_path.stem}_compressed{input_path.suffix}"),
            ]
        )
        os.unlink(output_path / current_name)
        current_name = f"{input_path.stem}_compressed{input_path.suffix}"

    os.rename(output_path / current_name, output_path / output_name)
    return to_test


def main() -> None:
    args = parse_args()

    if args.input.is_dir():
        inputs = args.input.glob("*.*")
    else:
        inputs = [args.input]

    for input_file in inputs:
        print(f"Processing {input_file}")
        get_predictions(input_file, args.output, compress=args.compress)


if __name__ == "__main__":
    main()
