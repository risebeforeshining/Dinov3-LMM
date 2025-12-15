#!/bin/bash

WANDB_PROJECT="EndoDinov3-SFT"
WANDB_RUN_ID="vicuna7b_dinov3_sft_cholec_train_2"
export WANDB_PROJECT WANDB_RUN_ID

deepspeed --include localhost:4,5 \
    dinov3/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --version v1 \
    --data_path /dataT0/Free/wcfei/dataset/sft-cholec80-train.json \
    --image_folder /dataT0/Free/wcfei/dataset/cholec80 \
    --vision_tower dinov3 \
    --pretrain_mm_mlp_adapter ./checkpoints/endo-dinov3-pretrain-vicuna7b/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --unfreeze_mm_vision_tower True \
    --mm_vision_tune_layers 2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_vision_lr 5e-6 \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/dinov3-v1.5-7b-cholec17k \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
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
