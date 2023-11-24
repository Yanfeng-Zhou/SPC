# Spatial and Planar Consistency for Semi-Supervised Volumetric Medical Image Segmentation

This is the official code of [Spatial and Planar Consistency for Semi-Supervised Volumetric Medical Image Segmentation](https://proceedings.bmvc2023.org/84/) (BMVC 2023).

## Overview
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SPC/blob/main/figure/Overview.png" width="100%" ></img>
<br>Architecture of LHCC.
</p>

## Quantitative Comparison

Comparison with state-of-the-art models on LA and P-CT test set. <font color="Red">**Red**</font> and **bold** indicate the best and second best performance.
<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SPC/blob/main/figure/Comparison%20results.png" width="100%" >
</p>

## Qualitative Comparison

<p align="center">
<img src="https://github.com/Yanfeng-Zhou/SPC/blob/main/figure/Qualitative%20results.png" width="100%" >
<center>Qualitative results on LA (left) and P-CT (right). (a) MT. (b) DTC. (c) MC-Net. (d) MC-Net+. (e) SPC. (f) Ground truth. The green arrows highlight the difference among of the results.</center>
</p>

## Requirements
```
albumentations==0.5.2
mayavi==4.8.1
MedPy==0.4.0
numpy==1.18.5
opencv_python==4.2.0.32
Pillow==9.5.0
scikit_image==0.19.1
scikit_learn==1.2.2
scipy==1.4.1
SimpleITK==2.2.1
skimage==0.0
torch==1.8.0
torchio==0.18.53
torchvision==0.9.0
visdom==0.1.8.9
```
## Usage
**Data preparation**
Your datasets directory tree should be look like this:
```
dataset
├── train_sup_20
    ├── image
        ├── 1.tif
        ├── 2.tif
        └── ...
    └── mask
        ├── 1.tif
        ├── 2.tif
        └── ...
├── train_unsup_80
    ├── image
├── val
    ├── image
    └── mask
```
**Training**
```
python -m torch.distributed.launch --nproc_per_node=4 train_semi_SPC.py
```
**Testing**
```
python -m torch.distributed.launch --nproc_per_node=4 test_SPC.py
```

## Citation
If our work is useful for your research, please cite our paper:
```
@inproceedings{Zhou_2023_BMVC,
author    = {Yanfeng Zhou and yiming huang and Ge Yang},
title     = {Spatial and Planar Consistency for Semi-Supervised Volumetric Medical Image Segmentation},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0084.pdf}
}
```
