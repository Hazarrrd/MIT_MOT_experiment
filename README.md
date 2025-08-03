URUCHOMIENIE:
W terminalu CMD wpisać kolejno komendy:
cd  D:\Desktop\MIT_MOT_experiment
D:
mitmot\Scripts\activate
python run.py
............

##demo kamerki live 2D:
python3 pose_estimation/inferencer_demo.py -i webcam -o x -s

##demo kamerki live 3D:
/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/body3d_img2pose_demo.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input webcam --show


.....................................
## model 3 d w wizalizacji 2d
/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/pose_inferencer_with_tracker.py --det_config /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py --det_checkpoint /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth --pose_config /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288.py --pose_checkpoint /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmw3d-l_8xb64_cocktail14-384x288-794dbc78_20240626.pth --input webcam --show

## demo modelu 2d w wizualizacji 2d topdown
/home/janek/miniconda3/envs/bundesliga/bin/python pose_estimation/pose_inferencer_with_tracker.py --det_config /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_640-8xb32_coco-person.py --det_checkpoint /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/det_models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth --pose_config /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmpose-l_8xb256-420e_coco-384x288.py --pose_checkpoint /home/janek/numlabs/repozytoria/ext_2024_MOT_and_pose_research/models/rtmpose/rtmpose-l_simcc-body7_pt-body7_420e-384x288-3f5a1437_20230504.pth --input webcam --show

...
wizaluzacja

python postprocessing/elbow_metric_comphresive.py   --json /home/janek/psychologia/MIT_MOT_experiment/results_real/experiment_2025-06-07-10-05-45_test/inference_results/3d/results_MIT_block_2_trial1.json   --video /home/janek/psychologia/MIT_MOT_experiment/results_real/experiment_2025-06-07-10-05-45_test/inference_results/3d/MIT_block_2_trial1.avi  --out_video /home/janek/psychologia/MIT_MOT_experiment/results_real/experiment_2025-06-07-10-05-45_test/inference_results/3d/MIT_block_2_trial1_angles.mp4   --out_csv /home/janek/psychologia/MIT_MOT_experiment/results_real/experiment_2025-06-07-10-05-45_test/inference_results/3d/MIT_block_2_trial1_angles.csv