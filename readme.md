# Compensating Distribution Drifts in Continual Learning with Pre-trained Vision Transformers

## Abstract
Recent works have evidenced how a sequential fine-tuning (SeqFT) phase of pre-trained vision transformers (ViTs) followed by a classifier refinement process through approximate distributions of class features, offers effective solutions to class incremental learning (CIL). However, this approach suffers from distribution drift due to the sequential optimization of shared backbone parameters, leading to a mismatch between the approximate distributions of previous classes and those of the updated model. This distribution mismatch generally leads to degraded performance in classifier refinement over time. To tackle this issue, we introduce the latent space transition operator, built on which we propose the Sequential Learning with Drift Compensation (SLDC) method. First, the linear SLDC method, which estimates a linear operator, is developed by solving a regularized least-squares problem between pre- and post-optimization features. Hereafter, the weak-nonlinear SLDC method, which assumes that appropriate transition operators are located at the intersection between linear and nonlinear regions, is developed by constructing learnable weak-nonlinear transformations. Finally, in both variants, knowledge distillation (KD) is applied to further mitigate the representation drift. Extensive experiments on CIL benchmarks demonstrate that SLDC significantly enhances the performance of SeqFT. Notably, by combining KD (to reduce representation drift) with SLDC (to counteract distribution drift), SeqFT achieves comparable performance to joint training across all evaluated datasets.

## Dataset Preparation
To set up the datasets for this project, follow these steps:

- Create a directory named `datasets` within your project workspace (e.g., `./datasets/`).
- **CIFAR-100**: This dataset is automatically downloaded upon running the training script, requiring no manual setup.
- **ImageNet-R**, **Cars-196**, **CUB-200**: Obtain the pre-processed versions of these datasets from the [PILOT: A Pre-Trained Model-Based Continual Learning Toolbox](https://github.com/sun-hailong/LAMDA-PILOT) repository. After downloading, place them into the `datasets` directory.

Ensure all datasets are properly positioned in the `datasets` folder before initiating any experiments to avoid runtime issues.

## Environment Requirements
The project requires the following Python packages with specific versions:

- `torch==1.12.0`
- `torchvision==0.13.0`
- `timm==0.5.4`
- `tqdm`
- `numpy`
- `scipy`
- `quadprog`
- `POT`

Make sure your environment matches these versions to ensure compatibility.
## System Configuration
This code is implemented in PyTorch and is designed to run on a single **NVIDIA GeForce RTX 4090** GPU. The experiments were conducted using four RTX 4090 GPUs, with each GPU utilized individually for separate runs (i.e., no multi-GPU parallelism within a single experiment). The system configuration is as follows:

- **Framework**: PyTorch
- **GPU**: NVIDIA GeForce RTX 4090 (4 GPUs, each used independently)
- **Operating System**: Linux 

Ensure your system has the appropriate NVIDIA drivers and CUDA toolkit installed (e.g., CUDA 11.8 or compatible) to support the RTX 4090.

## Training Instructions
To reproduce the experiments, use the following commands. Each command trains the model on a specific dataset with the Sequential Learning with Drift Compensation (SLDC) method. 


### ImageNet-R :
```
python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0
```
Alternative configuration:
```
python3 main_sldc.py --dataset="imagenet-r" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0

```


### CIFAR100 :
```
python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0

```
Alternative configuration:
```
python3 main_sldc.py --dataset="cifar100_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0

```
### Cars-196 :
```
python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0


```
Alternative configuration:
```
python3 main_sldc.py --dataset="cars196_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0


```
### CUB-200 :
```
python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora-mocov3" --gamma_norm=0.1 --gamma_kd=1.0

```
Alternative configuration:
```
python3 main_sldc.py --dataset="cub200_224" --convnet_type="vit-b-p16-lora" --gamma_norm=0.1 --gamma_kd=1.0

```
## Notes

The `--convnet_type` parameter specifies the Vision Transformer backbone. Options include:

- `vit-b-p16-lora-mocov3`: MoCo v3 pre-trained with LoRA
- `vit-b-p16-lora`: Standard LoRA

## Acknoledgements
We would like to express our sincere gratitude to the [PyCIL](https://github.com/G-U-N/PyCIL) project for their foundational contributions, which have greatly influenced the development of this repository. Their work has been invaluable in advancing our research and implementation.