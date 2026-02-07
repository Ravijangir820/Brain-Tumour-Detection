# 🧠 Brain Tumor Classification using Deep Learning

**Capstone Project -- PyTorch**

## Overview

This project implements an end-to-end deep learning pipeline for brain
MRI tumor classification using PyTorch. It includes training multiple
CNN architectures, transfer learning with ResNet50, fine-tuning,
detailed evaluation, and explainable AI using Grad-CAM.

## Classes

-   brain_glioma
-   brain_menin
-   brain_tumor

## Models

1.  **VAF** -- Lightweight custom CNN baseline\
2.  **UNET-inspired CNN** -- Encoder-style CNN for classification\
3.  **CustomCNN** -- Deeper handcrafted CNN with dropout\
4.  **ResNet50** -- ImageNet pretrained + fine-tuning

## Training Features

-   Mixed Precision Training (AMP)
-   ReduceLROnPlateau LR scheduler
-   Early stopping
-   CSV logging (loss, accuracy, LR)
-   Checkpointing best models
-   Windows-safe multiprocessing

## Evaluation Features

-   Classification report
-   Confusion matrix
-   ROC--AUC curves
-   Grad-CAM heatmaps
-   Misclassified image export
-   TensorBoard visualization

## Project Structure

See README in repository root for full directory layout.

## How to Run

### Train

``` bash
python train_pytorch.py
```

### Evaluate

``` bash
python evaluate.py
```

### TensorBoard

``` bash
tensorboard --logdir tensorboard_eval
```

## Author

Ravi Jangid\
Integrated M.Tech (CSE), VIT-AP University
