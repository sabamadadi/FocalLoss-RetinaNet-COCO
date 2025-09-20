# Focal Loss for Dense Object Detection – Implementation

## 🎥 Presentation Videos (Persian)
- [Video Presentation of the Paper (in Persian)]([your-paper-video-link-here](https://drive.google.com/file/d/16kOf-uzZvi2_cymJs5GuorxXMTedl5zZ/view))  
- [Video Presentation of the Implementation/Code (in Persian)]([your-code-video-link-here](https://drive.google.com/file/d/13SMjwUFaWaVpJg_r1CFTHL1cmngxuAPg/view))  

---

## 📖 Introduction
This repository presents an implementation of the **Focal Loss for Dense Object Detection** paper by *Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár (FAIR, 2017)*.  
The paper introduces **Focal Loss**, a novel loss function designed to address the extreme foreground-background class imbalance in dense object detection. The method is applied to the **RetinaNet** architecture, which achieves state-of-the-art accuracy on challenging benchmarks such as **COCO**, while maintaining efficiency.

## 🧑‍💻 My Work
In this project, I studied the paper and implemented the proposed method using **PyTorch** and the **COCO 2017 dataset**.  
I implemented:
- A **custom Focal Loss** class in PyTorch.  
- A RetinaNet-inspired architecture with a **ResNet backbone** and **Feature Pyramid Network (FPN)**.  
- Training and evaluation loops with COCO dataset integration.  

Although the code is complete, due to **server unavailability** and the **computationally expensive training process**, I was unable to fully run the model on my personal laptop. Nevertheless, the pipeline is implemented and ready for execution in a GPU-enabled environment.

## 📂 Dataset
The project uses the **COCO 2017 dataset**:  
- Training images: `train2017` (118k images)  
- Validation images: `val2017` (5k images)  
- Annotations: `instances_train2017.json`, `instances_val2017.json`  

## 🛠️ Implementation
The code includes:
- **Data loading** with COCO annotations using `torchvision.datasets.CocoDetection`.  
- **Custom Focal Loss** implemented from scratch.  
- **Backbone** based on ResNet-50.  
- **Feature Pyramid Network (FPN)** for multi-scale feature representation.  
- **Subnets** for classification and box regression.  
- Training and evaluation functions with `tqdm` progress bars.  

## 🚧 Limitations
Due to the computational cost of training RetinaNet on COCO, I was unable to finish a full training run on my laptop. This work demonstrates the methodology and implementation, which can be executed successfully in a GPU server or cloud environment.  
---
