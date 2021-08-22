# Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation

This is an Pytorch implementation of "Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation" accepted by CVPR2021.
More details of this work can be found in our paper: [[Paper (arxiv)]](https://arxiv.org/abs/2104.09415).

The code is based on [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) implementation.

## Install and Data preparation

Refer to [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME).

## Training
To run training on DomainNet in the 3-shot scenario using resnet34,

`CUDA_VISIBLE_DEVICES=gpu_id python main.py --dataset multi --source real --target sketch --net resnet34 --num 3 --lr_f 1.0 --multi 0.1`

where, gpu_id = 0,1,2,3...

### Reference
If you consider using this code or its derivatives, please consider citing:

```
@InProceedings{li2021cross,
    author    = {Li, Jichang and Li, Guanbin and Shi, Yemin and Yu, Yizhou},
    title     = {Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2505-2514}
}
```
