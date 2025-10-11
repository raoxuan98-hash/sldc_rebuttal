#!/usr/bin/env bash
set -euo pipefail

# ==============================================
# 多GPU DS-AL 实验批运行脚本（含 fsa_steps 三方案）
# ==============================================

FSA_STEPS=(0 400 1200)

# ---------- 第一批 ----------
for FSA in "${FSA_STEPS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python3 main_acil.py \
    --dataset="imagenet-r" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=4 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=1 python3 main_acil.py \
    --dataset="cifar100_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=10 \dsal_0624_logs_authors/imagenet-r_vit-b-p16-mocov3/init-20_inc-4_opt-adamw_lr-0.0001_wd-3e-05_lora-basic_lora_rank-4_epoch-1_seed-1993_mainact-relu_compact-tanh_fusion-1_fsa-1200-lr-0.0001
    --increment=2 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=2 python3 main_acil.py \
    --dataset="cars196_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=4 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=4 python3 main_acil.py \
    --dataset="cub200_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=4 \
    --fsa_steps=$FSA &
  sleep 10
done

wait

# ---------- 第二批 ----------
for FSA in "${FSA_STEPS[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python3 main_acil.py \
    --dataset="imagenet-r" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=2 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=1 python3 main_acil.py \
    --dataset="cifar100_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=10 \
    --increment=1 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=2 python3 main_acil.py \
    --dataset="cars196_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=2 \
    --fsa_steps=$FSA &
  sleep 10

  CUDA_VISIBLE_DEVICES=4 python3 main_acil.py \
    --dataset="cub200_224" \
    --vit_type="vit-b-p16-mocov3" \
    --model_name="dsal" \
    --init_cls=20 \
    --increment=2 \
    --fsa_steps=$FSA &
  sleep 10
done

wait

echo "✅ 所有 DS-AL 实验 (fsa_steps=0,400,1200) 均已完成！"


