# Conv-MLP：A Convolution and MLP Mixed Model for Multi-Modal Face Anti-Spoofing




## Introduction

This repo is a python reimplementation of Conv-MLP.

Conv-MLP is a simple yet effective  architecture for multi-modal face anti-spoofing, which incorporates local patch convolution with global MLP. 

Conv-MLP breaks the inductive bias limitation of traditional full CNNs and can be expected to better exploit long-range dependencies.

![alt text](image/conv-mlp-1-1118.png )
The overall pipeline of Conv-MLP.


## Performace

We conduct experiments to evaluate the performance of
Conv-MLP in terms of accuracy and computational efficiency
on multi-modal benchmarks (WMCA and CeFA) in comparison with existing state-of-the-art methods, including full CNN models and transformer-based ViT models.

### Results on WMCA

![alt text](image/res-wmca.png )
As shown, Conv-MLP ranks first in terms of the mean ACER on the seven unseen protocols (7.16 $\pm$ 11.10%), which implies Conv-MLP can extract discriminative representations and performs well on unseen scenarios.

![alt text](image/grad-wmca.png )
The visualized gradient map of typical samples from WMCA dataset.




## Get Started

### Package Requirement

- Python 3.7
- torch 1.6.0
- opencv-python, numpy, shutil, torchvision, tqdm

### Datasets

We train and test on the [WMCA](https://www.idiap.ch/dataset/wmca) and [CeFA](https://sites.google.com/qq.com/face-anti-spoofing/dataset-download/casia-surf-cefacvpr2020) datasets respectively. According to the usage license agreement, we do not have the right to provide the datasets in public. If you need to use them, please refer to the link and apply to the relevant scientific institutions for research usage.

### Training

- Starting from scratch
```
python train.py
```
- Pretraining
```
python train.py --pretrained_model='model_name.pth'
```
Note that for every training, you need to goto `./data/prepare_data.py` and modify the corresponding data path.

You can also find variables, such as *batch_size*, *patch_size*, *learning rate*, and *number of epoches* in the `train.py`.

### Evaluation

```
python train.py --mode='infer_val' --pretrained_model='model_name.pth'
```
Note that for every testing, you also need to goto `./data/prepare_data.py` and modify the corresponding data path.

## Pretrained Models

We provide pre-trained models on two datasets separately.

- WMCA

    *BaiduCloud:*

    *GoogleDrive:*
- CeFA

    *BaiduCloud:*

    *GoogleDrive:*

<!-- ## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```

@article{guo2021sample,
  title={Sample and Computation Redistribution for Efficient Face Detection},
  author={Guo, Jia and Deng, Jiankang and Lattas, Alexandros and Zafeiriou, Stefanos},
  journal={arXiv preprint arXiv:2105.04714},
  year={2021}
}

@inproceedings{an2020partical_fc,
  title={Partial FC: Training 10 Million Identities on a Single Machine},
  author={An, Xiang and Zhu, Xuhan and Xiao, Yang and Wu, Lan and Zhang, Ming and Gao, Yuan and Qin, Bin and
  Zhang, Debing and Fu Ying},
  booktitle={Arxiv 2010.05222},
  year={2020}
}

@inproceedings{deng2020subcenter,
  title={Sub-center ArcFace: Boosting Face Recognition by Large-scale Noisy Web Faces},
  author={Deng, Jiankang and Guo, Jia and Liu, Tongliang and Gong, Mingming and Zafeiriou, Stefanos},
  booktitle={Proceedings of the IEEE Conference on European Conference on Computer Vision},
  year={2020}
}

@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}

@inproceedings{guo2018stacked,
  title={Stacked Dense U-Nets with Dual Transformers for Robust Face Alignment},
  author={Guo, Jia and Deng, Jiankang and Xue, Niannan and Zafeiriou, Stefanos},
  booktitle={BMVC},
  year={2018}
}

@article{deng2018menpo,
  title={The Menpo benchmark for multi-pose 2D and 3D facial landmark localisation and tracking},
  author={Deng, Jiankang and Roussos, Anastasios and Chrysos, Grigorios and Ververas, Evangelos and Kotsia, Irene and Shen, Jie and Zafeiriou, Stefanos},
  journal={IJCV},
  year={2018}
}

@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
``` -->
