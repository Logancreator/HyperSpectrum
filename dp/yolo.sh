#!/bin/bash
#SBATCH --job-name=YOLO
#SBATCH --partition=gpu-3090,gpu-v100s
#SBATCH --ntasks-per-node=8
#SBATCH --output=%j.log
#SBATCH --gpus=1
source activate /public/home/changjianye/anaconda3/envs/deeplearning/
cd /public/home/changjianye/project/phenomics/yolov5-master
export LD_LIBRARY_PATH="/public/home/changjianye/anaconda3/envs/deeplearning/lib:$LD_LIBRARY_PATH"
python=/public/home/changjianye/anaconda3/envs/deeplearning/bin/python

# $python train.py \
# 	--cfg  models/yolov5s.yaml \
# 	--data data/maize.yaml \
# 	--epoch 200 --batch-size 8 --img 384


$python detect.py --weights runs/train/exp6/weights/best.pt --source data/images