# water-segmentation

This repository contains code and resources for [ISAE Supaero's SDD hackathon 2024](https://supaerodatascience.github.io/hackathon.html). 

The goal of this project is to develop an algorithm that can accurately segment water regions in images. This can be useful in various applications such as environmental monitoring, flood prediction, and satellite imagery analysis. 

The evaluation metric used for the model's performance was mIOU.

![Mask example](/utils/images/mask_example.png)

## Solutions

Several different architectures and methods were tested during the course of the hackathon. Below is a list of different architectures, encoders, loss functions and Augmentations tested.

### Architectures

- [U-Net](https://arxiv.org/abs/1505.04597)
- [U-Net++](https://arxiv.org/pdf/1807.10165v1.pdf)
- [DeepLabV3](https://arxiv.org/abs/1706.05587)
- [FPN](https://arxiv.org/pdf/1612.03144v2.pdf)

### Encoders

- ResNet (18, 50, 101 and 152)
- EfficientNet
- ResNeXt

Models with both random and pre-trained weights were tested.

### Loss functions

- Dice Loss
- Jaccard Loss
- BCE

### Augmentations

Not taking into consideration the common augmentations (crops, shifts, rotates, jitters, etc.), we observed a great improvement in the model's performance by using [Histogram Matching](https://en.wikipedia.org/wiki/Histogram_matching). 

We conjecture that this is due to the fact that the dataset often had examples with out-of-distribution conditions of lightning, camera type, and vegetation. Applying a histogram matching of these images allowed the model to generalize better to all types of different conditions. 

## Results

![Results](/utils/images/results_on_testset.png)