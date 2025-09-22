#!/bin/bash

FED_MODEL="FedMAE"
cd /home/erenk/Documents/USYD/Honours/SSFL/code/${FED_MODEL}/

DATASET="Retina"
DATA_PATH="/home/erenk/Documents/USYD/Honours/SSFL/data/${DATASET}/"

SPLIT_TYPE="split_3"
N_CLASSES=2
N_CLIENTS=5
MASK_RATIO=0.6
N_GPUS=1

# ---------- FedMAE pretraining ----------
EPOCHS=1600
BLR="2e-3"
BATCH_SIZE=32
TOPK_RATIO=0.1

OUTPUT_PATH="/home/erenk/Documents/USYD/Honours/SSFL/data/ckpts/${DATASET}/${FED_MODEL}/pretrained_epoch${EPOCHS}_${SPLIT_TYPE}_blr${BLR}_bs${BATCH_SIZE}_ratio${MASK_RATIO}_dis${N_GPUS}"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=${N_GPUS} run_pretraining.py \
    --train_mode pretrain \
    --data_path ${DATA_PATH} \
    --dataset ${DATASET} \
    --output_dir ${OUTPUT_PATH} \
    --blr ${BLR} \
    --batch_size ${BATCH_SIZE} \
    --save_ckpt_freq 200 \
    --max_communication_rounds ${EPOCHS} \
    --split_type ${SPLIT_TYPE} \
    --mask_ratio ${MASK_RATIO} \
    --model mae_vit_base_patch16 \
    --warmup_epochs 40 \
    --weight_decay 0.05 \
    --norm_pix_loss \
    --sync_bn \
    --n_clients ${N_CLIENTS} \
    --E_epoch 1 \
    --num_local_clients -1 \
    --topk_ratio ${TOPK_RATIO}

# ---------- Finetuning with pretrained model ----------
FT_EPOCHS=100
FT_LR="3e-3"
FT_BATCH_SIZE=64
CKPT_PATH="${OUTPUT_PATH}/checkpoint-1599.pth"

OUTPUT_PATH_FT="${OUTPUT_PATH}/finetune_${DATASET}_epoch${FT_EPOCHS}_${SPLIT_TYPE}_lr${FT_LR}_bs${FT_BATCH_SIZE}_dis${N_GPUS}"

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 torchrun --nproc_per_node=${N_GPUS} run_finetuning.py \
    --train_mode finetune \
    --data_path ${DATA_PATH} \
    --dataset ${DATASET} \
    --finetune ${CKPT_PATH} \
    --nb_classes ${N_CLASSES} \
    --output_dir ${OUTPUT_PATH_FT} \
    --save_ckpt_freq 50 \
    --model vit_base_patch16 \
    --batch_size ${FT_BATCH_SIZE} \
    --split_type ${SPLIT_TYPE} \
    --blr ${FT_LR} \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --reprob 0.25 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --n_clients ${N_CLIENTS} \
    --E_epoch 1 \
    --max_communication_rounds ${FT_EPOCHS} \
    --num_local_clients -1
