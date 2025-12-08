#!/bin/bash

WANDB_PROJECT="EndoDinov3-Pretrain"
WANDB_RUN_ID="vicuna7b_dinov3_mlp_align"
export WANDB_PROJECT WANDB_RUN_ID

deepspeed --include localhost:4,5 \
    dinov3/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version plain \
    --data_path /dataT0/Free/wcfei/dataset/train.json \
    --image_folder /dataT0/Free/wcfei/dataset/PSI-AVA/keyframes \
    --vision_tower dinov3 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --bf16 True \
    --output_dir ./checkpoints/endo-dinov3-pretrain-vicuna7b \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --lazy_preprocess True \
    --report_to wandb
