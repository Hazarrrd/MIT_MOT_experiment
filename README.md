URUCHOMIENIE:
W terminalu CMD wpisać kolejno komendy:
cd  D:\Desktop\MIT_MOT_experiment
D:
mitmot\Scripts\activate
python run.py
............


uv pip install pip
uv pip install chumpy --no-build-isolation

/home/janek/miniconda3/envs/bundesliga/bin/python3 /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/inferencer_demo.py --input webcam --output x.mp4

/home/janek/miniconda3/envs/bundesliga/bin/python /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input webcam --show --show-interval 0

/home/janek/miniconda3/envs/bundesliga/bin/python /home/janek/numlabs/repozytoria/mmpose/demo/topdown_demo_with_mmdet.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw-x_8xb320-270e_cocktail14-384x288.py /home/janek/.cache/torch/hub/checkpoints/rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth --input webcam --show --show-interval 0

###Final
/home/janek/miniconda3/envs/bundesliga/bin/python body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input webcam --show