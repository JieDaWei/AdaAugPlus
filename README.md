# AdaAug+ in Pytorch

## Introduction
This project implements the method proposed in the paper "ADAAUG+: A Reinforcement Learning-Based Adaptive Data Augmentation for Change Detection. (TGRS 2024) [paper](https://ieeexplore.ieee.org/document/10714383)" The method utilizes reinforcement learning for adaptive data augmentation to enhance the performance of change detection tasks.

## Table of Contents
- [Dataset Preparation](#dataset-preparation)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)

## Dataset Preparation
### Data Structure
    dataset/  
    ├── A/  
    ├── B/
    ├── label/  
    └── list/
        ├── train.txt: train images  
        ├── test.txt: test images
        └── trainc.txt: strategy evaluation images (contains at least one change)
### Data Download
1. **BCD** (https://study.rsgis.whu.edu.cn/pages/download/building_dataset.html)
2. **LEVIR** (https://justchenhao.github.io/LEVIR/)
3. **GZCD** (https://github.com/daifeng2016/Change-Detection-Dataset-for-High-Resolution-Satellite-Imagery)
4. **EGYBCD** (https://github.com/oshholail/EGY-BCD)


## Training the Model
To train the model, run the following command:

    python main_cd.py

## Evaluating the Model

    python eval_cd.py


## 4. Cite
If you find this repo useful for your research, please cite our paper:

    @ARTICLE{10714383,
    author={Huang, Rui and Wei, Jieda and Xing, Yan and Guo, Qing},
    journal={IEEE Transactions on Geoscience and Remote Sensing}, 
    title={AdaAug+: A Reinforcement Learning-Based Adaptive Data Augmentation for Change Detection}, 
    year={2024},
    volume={62},
    number={},
    pages={1-12},
    doi={10.1109/TGRS.2024.3478218}}
