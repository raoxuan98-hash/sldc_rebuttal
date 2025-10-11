CUDA_VISIBLE_DEVICES=0 python3 main_sldc.py --dataset="imagenet-r" --vit_type="vit-b-p16-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=1 python3 main_sldc.py --dataset="cars196_224" --vit_type="vit-b-p16-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=2 python3 main_sldc.py --dataset="cifar100_224" --vit_type="vit-b-p16-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
sleep 45
CUDA_VISIBLE_DEVICES=4 python3 main_sldc.py --dataset="cub200_224" --vit_type="vit-b-p16-mocov3" --gamma_norm=0.1 --gamma_kd=1.0 --smart_defaults --gamma_2=0.1&
wait

