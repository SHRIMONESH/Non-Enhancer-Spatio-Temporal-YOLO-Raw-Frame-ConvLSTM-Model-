# Non-Enhancer Spatio-Temporal YOLO (NEST-YOLO)
### Raw-Frame ConvLSTM Detection for Nighttime Video

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-orange)
![Architecture](https://img.shields.io/badge/Arch-ConvLSTM%2BAttention-purple)
![Preprocessing](https://img.shields.io/badge/Preprocessing-None-red)

## 📖 Overview

**Non-Enhancer Spatio-Temporal YOLO (NEST-YOLO)** is a purely data-driven object detection model designed for nighttime video sequences. 



Unlike traditional low-light models, NEST-YOLO **does not use any image enhancement** (no CLAHE, Gamma correction, or denoising). Instead, it relies entirely on temporal cues and memory to learn invariant representations directly from raw, noisy frames. By aggregating features across time using ConvLSTM and Attention, the model filters out random noise and light fluctuations, providing stable detection without the computational overhead of preprocessing pipelines.

---

## 🚀 Key Features

* **Zero Preprocessing Overhead:** Removes the need for computationally expensive enhancement algorithms (like CLAHE), simplifying the runtime pipeline and reducing latency.
* **Temporal Robustness (ConvLSTM):** Uses a memory mechanism to reconstruct structural information from sequence history, even when individual frames are pitch black or corrupted by headlights.
* **Flicker-Free Detection:** Because the model relies on temporal consistency rather than frame-by-frame pixel manipulation, it avoids the "flickering" often caused by adaptive enhancement techniques.
* **Sparsity-Regularized Attention:** Incorporates lightweight attention mechanisms with sparsity regularization, forcing the network to focus only on relevant spatial regions and ignore background noise.
* **Multi-Branch Fusion:** Adaptively fuses spatial features to handle varying object scales in raw lighting conditions.

---

## 🏗️ Architecture

The model pipeline is designed to accept raw video sequences and output stable detections.
<img width="2048" height="2048" alt="app3 (2)" src="https://github.com/user-attachments/assets/66819158-e15d-40b9-955a-e41368fa1517" />
<img width="2048" height="2048" alt="sap3" src="https://github.com/user-attachments/assets/86fa57d5-44b9-418e-9494-8b23552be5e8" />

```mermaid
graph TD
    Input[Raw Video Sequence] --> Backbone[YOLO Backbone (Frame-wise)]
    Backbone --> Feat[Multi-Scale Features]
    
    subgraph "Temporal Aggregation"
    Feat --> CLSTM[ConvLSTM Layers]
    CLSTM -->|Temporal Context| Attn[Sparsity-Based Attention]
    end
    
    subgraph "Fusion & Detection"
    Attn --> Fusion[Multi-Branch Fusion]
    Fusion --> Head[Detection Head]
    Head --> Output[Stable Bounding Boxes]
    end
