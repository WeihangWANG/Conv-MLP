# Conv-MLP：A Convolution and MLP Mixed Model for Multi-Modal Face Anti-Spoofing (RGB Branch)

## Introduction
This is the branch for the implementation of Conv-MLP in RGB mode. We give the python codes on the OULU-NPU dataset for example.

## Get Started

### Package Requirement
- Python 3.7
- torch, torchvision
- opencv-python, numpy, shutil, tqdm, timm, imgaug
You can also install `apex` to streamline mixed precision and distributed training in Pytorch. Please visit (https://github.com/NVIDIA/apex) for more details. If you donot want to use `apex`, just comment out the relevent codes in `main.py`.

### Datasets
We train and test on the [OULU-NPU](https://sites.google.com/site/oulunpudatabase/) dataset. According to the usage license agreement, we do not have the right to provide the datasets in public. If you need to use them, please refer to the link and apply to the relevant scientific institutions for research usage.

In our work, we crop the face region from the scene image using [insightface](https://github.com/deepinsight/insightface).

### Training
```
python -m torch.distributed.launch --nproc_per_node 4 --master_port 12345  main.py \
--cfg configs/config.yaml --data-path './oulu-npu' --batch-size 128
```
Note: For each protocol of OULU-NPU, you need to modify the data path in `/data/prepare_data.py` and adjust the revelent hyper-parameters in `/data/augmentaion_oulu.py`.


