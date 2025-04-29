# Bangla Food Image Classification Using Vision Transformer Model

This project implements a Vision Transformer (ViT)-based deep learning model for classifying Bangladeshi food images. The model leverages the self-attention mechanism of transformer architecture for high-accuracy image classification.

## 📌 Project Objective

To classify images of Bangladeshi food using a Vision Transformer (ViT) architecture and evaluate its performance on a labeled dataset. The model is trained using TensorFlow and Keras with image augmentation and evaluated for generalization on unseen test data.

---

## 📁 Dataset

The dataset is structured into three directories:
- `/train`: Training images
- `/val`: Validation images
- `/test`: Testing images

Each directory contains subfolders named after the food categories (e.g., `biryani`, `pitha`, `hilsa`, etc.), and each subfolder contains the corresponding images.

> Dataset Path (used in code):  
- `train_image_path = "/kaggle/input/fppd03/train"`  
- `valid_image_path = "/kaggle/input/fppd03/val"`  
- `test_image_path = "/kaggle/input/fppd03/test"`

---

## 🧠 Model Architecture

This implementation uses a custom Vision Transformer (ViT) with the following configuration:
- **Image Size**: 128×128
- **Patch Size**: 16
- **Projection Dimension**: 64
- **Transformer Layers**: 8
- **Number of Attention Heads**: 4
- **MLP Head Units**: [2048, 1024]
- **Dropout**: Applied in MLP and representation layers

The model follows the general ViT pipeline:
1. Extract patches from input image.
2. Linearly project each patch and add positional embedding.
3. Pass through transformer encoder blocks (multi-head attention + feedforward layers).
4. Apply global average pooling.
5. Pass through a multilayer perceptron (MLP).
6. Output softmax predictions over food classes.

---
## 🚀 Training Instructions
The model is compiled and trained with:

**Optimizer: Adam (lr=1e-4)**

**Loss Function: Categorical Crossentropy**

**Metrics: Accuracy**

**Epochs: 500**

**Batch Size: 32**

## 📌 Results
The Vision Transformer model provides high performance on Bangla food classification. Its attention-based architecture allows it to focus on relevant parts of the food image, making it suitable for complex image classification tasks.

## 📎 Acknowledgements
TensorFlow & Keras for deep learning framework

Kaggle for dataset hosting and GPU compute

Vision Transformer (ViT) architecture by Google Research

## 🔧 Setup & Dependencies

Make sure you have the following libraries installed:

```bash
pip install tensorflow numpy matplotlib
