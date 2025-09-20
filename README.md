# Focal Loss for Dense Object Detection – Implementation

## 🎥 Presentation Videos (Persian)

▶️ **Paper Presentation (in Persian):** [Watch here](https://drive.google.com/file/d/16kOf-uzZvi2_cymJs5GuorxXMTedl5zZ/view)  

▶️ **Code Presentation (in Persian):** [Watch here](https://drive.google.com/file/d/13SMjwUFaWaVpJg_r1CFTHL1cmngxuAPg/view)  

---

## Abstract  
Dense object detection presents unique challenges due to the extreme imbalance between foreground and background classes. The seminal paper **“Focal Loss for Dense Object Detection”** (Lin et al., FAIR 2017) introduced the **Focal Loss** function, designed to address this imbalance by down-weighting easy negatives and focusing training on hard examples. The authors proposed the **RetinaNet** architecture, which achieved state-of-the-art results on COCO while maintaining a one-stage detector’s efficiency.  

In this report, I study the focal loss mechanism, reproduce its theoretical insights, and implement a RetinaNet-based object detection pipeline on the COCO dataset. My implementation integrates a custom **Focal Loss module**, a **Feature Pyramid Network (FPN)** backbone, and training/evaluation routines using PyTorch.

---

## Introduction  
Traditional one-stage detectors (e.g., SSD, YOLO) suffer from lower accuracy compared to two-stage detectors (e.g., Faster R-CNN) due to the extreme class imbalance: the number of background (easy negative) anchors dominates the relatively few foreground samples.  

The **Focal Loss** addresses this by modifying the standard cross-entropy loss:  

$$
FL(p_t) = - \alpha_t (1 - p_t)^\gamma \log(p_t)
$$

- **α**: balancing factor between positive/negative examples.  
- **γ**: focusing parameter, reducing the loss for well-classified examples.  
- As a result, the model emphasizes hard, misclassified examples during training.  

The **RetinaNet** architecture combines a **ResNet backbone with Feature Pyramid Networks (FPNs)** and dense prediction subnetworks for classification and bounding box regression.  

---

## Implementation  

### Dataset  
- **COCO 2017** dataset.  
- Used **train2017** for training and **val2017** for validation.  
- Implemented dataset loading with **`torchvision.datasets.CocoDetection`**.  
- Applied preprocessing: normalization with ImageNet means/stds.  

### Custom Focal Loss  
I implemented the focal loss in PyTorch, with tunable **α** and **γ** parameters. This function replaces binary cross-entropy for classification loss in dense detection.  

### Model Components  
1. **Backbone**: ResNet50 pretrained on ImageNet.  
2. **Feature Pyramid Network (FPN)**: Generates multi-scale feature maps (P3–P7).  
3. **Subnetworks**:  
   - **Classification head** with focal loss.  
   - **Regression head** for bounding box offsets.  
4. **Final Detector**: RetinaNet, built on top of backbone + FPN + subnets.  

### Training & Evaluation  
- Optimizer: **Adam**, learning rate = 1e-4.  
- Epochs: 10 (planned).  
- Batch size: 4.  
- Logging: progress tracked with **tqdm**.  
- Evaluation included per-batch predictions and cumulative loss tracking.  

---

## Results & Limitations  
- Due to **hardware constraints** (running on a local laptop and Kaggle kernels with limited resources), full training on the COCO dataset was **not computationally feasible**.  
- The experiment was **interrupted mid-training**, as the required GPU hours exceeded the available resources.  
- Nevertheless, the implementation demonstrates how focal loss integrates into RetinaNet and how COCO preprocessing, training loops, and evaluation are structured.  

---

## Conclusion  
This project highlights how **Focal Loss improves dense object detection** by addressing the imbalance problem in one-stage detectors. While I was not able to train RetinaNet fully due to server and computational limitations, the implementation faithfully follows the paper’s methodology and can be extended on more powerful hardware.  

Future improvements:  
- Training on distributed GPU clusters.  
- Hyperparameter tuning for focal loss (α, γ).  
- Comparing RetinaNet + Focal Loss with baseline detectors (SSD, Faster R-CNN).  

---

## Reference  
Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).  
**Focal Loss for Dense Object Detection.** ICCV 2017.  
[https://arxiv.org/abs/1708.02002](https://arxiv.org/abs/1708.02002)
