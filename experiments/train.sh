# !/bin/bash

export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
LOG_FILE="results/train.log"

nohup deepspeed --num_gpus=8 --master_port 23456 run_mlora.py training_args.json > $LOG_FILE 2>&1