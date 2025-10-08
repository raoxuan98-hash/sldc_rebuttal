#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
wait

CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=1.0&
sleep 45
CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=1.0&
sleep 45
CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=1.0&
sleep 45
CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=1.0&
wait

CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=2.0&
sleep 45
CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=2.0&
sleep 45
CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=2.0&
sleep 45
CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=2.0&
wait


# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# wait

# cars196_224 dataset
# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults &
# wait

# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# wait

# cars196_224 dataset

# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults &
# wait


# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# wait

# cars196_224 dataset
# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# wait

# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# wait

# cars196_224 dataset
# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.0 --gamma_kd=0.0 --smart_defaults --optimizer=sgd --lrate=1e-3 &
# wait


# CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --optimizer=sgd --lrate=1e-3 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --optimizer=sgd --lrate=1e-3 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --optimizer=sgd --lrate=1e-3 --smart_defaults &
# sleep 45
# CUDA_VISIBLE_DEVICES=3 python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0 --optimizer=sgd --lrate=1e-3 --smart_defaults &
# wait