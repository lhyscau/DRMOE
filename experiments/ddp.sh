# !/bin/bash

# Set environment variables
export WANDB_DISABLED=true
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# LOG_FILE="/root/lhy/backup/blc_loss_beta/MOELoRA-peft/results/train.log"
LOG_FILE="results/train.log"
# # Run the training script with accelerate
# accelerate launch /root/lhy/MOELoRA-peft/run_mlora.py arg_fun.json

# nohup deepspeed --num_gpus=8 --master_port 23456 run_mlora.py /root/lhy/MOELoRA-peft/arg_ds_train.json > $LOG_FILE 2>&1 &
# nohup deepspeed --num_gpus=4 --master_port 23456 run_mlora.py /root/lhy/MOELoRA-peft/arg_ds_train.json > $LOG_FILE 2>&1
nohup deepspeed --num_gpus=8 --master_port 23456 run_mlora.py debug_arg_ds_train.json > $LOG_FILE 2>&1

# deepspeed --num_gpus=8 --master_port 23456 run_mlora.py arg_ds_predict.json

# STEP_SIZE=397
# MAX_STEP=9925
# PEFT_PATHS=()
# OUTPUT_PATHS=()

# for (( step=$STEP_SIZE; step<=$MAX_STEP; step+=$STEP_SIZE ))
# do
#   PEFT_PATHS+=("saved/moelora/checkpoint-$step")
#   OUTPUT_PATHS+=("results/pred/ckpt-$step")
# done

# if [ ${#PEFT_PATHS[@]} -ne ${#OUTPUT_PATHS[@]} ]; then
#   echo "Error: The number of PEFT_PATHS and OUTPUT_PATHS must be equal."
#   exit 1
# fi

# LOG_FILE="/root/lhy/MOELoRA-peft/results/predict.log"
# # Iterate over each pair of PEFT_PATH and OUTPUT_PATH and run the deepspeed command
# for i in "${!PEFT_PATHS[@]}"; do
#   PEFT_PATH=${PEFT_PATHS[i]}
#   OUTPUT_PATH=${OUTPUT_PATHS[i]}
# #   LOG_FILE="/root/lhy/MOELoRA-peft/results/prediction_${i}.log"
  
#   echo "Running DeepSpeed with peft_path: $PEFT_PATH and output_path: $OUTPUT_PATH"
#   deepspeed --num_gpus=4 --master_port 23456 run_mlora.py \
#     --do_predict \
#     --test_file datasets/test.json \
#     --cache_dir datasets \
#     --overwrite_cache \
#     --prompt_column input \
#     --response_column target \
#     --model_name_or_path /root/lhy/model/bloom-1b4-zh \
#     --peft_path $PEFT_PATH \
#     --output_dir $OUTPUT_PATH \
#     --overwrite_output_dir \
#     --max_source_length 1024 \
#     --max_target_length 196 \
#     --per_device_eval_batch_size 32 \
#     --predict_with_generate \
#     --lora_name moelora \
#     --expert_num 8 \
#     --remove_unused_columns false \
#     >> $LOG_FILE 2>&1

#   python /root/lhy/MOELoRA-peft/results/evaluate.py "/root/lhy/MOELoRA-peft/${OUTPUT_PATH}/test_predictions.json" \
#   >> $LOG_FILE 2>&1
# done