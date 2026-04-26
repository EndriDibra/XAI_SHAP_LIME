# Explainable AI for Image Classification using SHAP and LIME

## Author
**Endri Dibra**

## Project Overview

This project demonstrates the practical use of **Explainable Artificial Intelligence (XAI)** for deep learning image classification systems.

Modern Convolutional Neural Networks (CNNs) often achieve high accuracy, but their internal decision-making process is frequently considered a **black box**. This project addresses that challenge by applying two of the most widely used XAI frameworks:

- **SHAP** (SHapley Additive exPlanations)  
- **LIME** (Local Interpretable Model-Agnostic Explanations)

The objective is to visually explain **which image regions contributed the most** to the model’s final classification decision.

---

## Core Idea

Given an input image, a pretrained CNN performs object recognition.  
Then, XAI techniques are used to highlight:

- Important pixels  
- Influential regions  
- Positive and negative contributing features  
- Areas driving prediction confidence  

This improves model transparency, interpretability, and trustworthiness.

---

## Technologies Used

### Programming Language
- Python

### Deep Learning Framework
- TensorFlow / Keras

### Pretrained CNN Model
- **MobileNetV2** trained on ImageNet

### Explainability Libraries
- **SHAP**
- **LIME**

### Supporting Libraries
- OpenCV  
- NumPy  
- Matplotlib  
- Scikit-Image  

---

## Workflow

### 1. Image Input & Preprocessing

The system:

- Loads the input image  
- Converts it to RGB format  
- Resizes it to `224x224`  
- Applies MobileNetV2 preprocessing pipeline  

---

### 2. Object Classification

The pretrained **MobileNetV2** model predicts the top object classes from the image.

Example output:

1. Object Label #1  
2. Object Label #2  
3. Object Label #3  
4. Object Label #4  
5. Object Label #5  

with confidence scores.

---

### 3. SHAP Explanation

Using **GradientExplainer**, SHAP estimates feature contribution scores for image pixels.

Output includes:

- Heatmap of influential regions  
- Positive / negative importance areas  
- Model sensitivity visualization  

This reveals *why* the CNN favored a specific prediction.

---

### 4. LIME Explanation

LIME perturbs image segments (superpixels) and observes prediction changes.

Output includes:

- Important object boundaries  
- Interpretable segmented regions  
- Local explanation of the top predicted class  

---

## Final Visualization

The project displays three side-by-side outputs:

1. **Original Image**  
2. **SHAP Explanation Heatmap**  
3. **LIME Explanation Map**  

This allows direct comparison between two XAI approaches.

---

## Main Script

### `XAI_SHAP_LIME.py`

Contains full implementation for:

- Image loading  
- CNN classification  
- SHAP explanation generation  
- LIME explanation generation  
- Visualization of results  

---

## Why This Project Matters

As AI systems become more common in critical domains such as:

- Healthcare  
- Autonomous Vehicles  
- Security  
- Robotics  
- Finance  

understanding *why* a model made a decision becomes just as important as the prediction itself.

This project demonstrates how explainability tools can improve:

- Transparency  
- Accountability  
- Debugging  
- Trust in AI systems  

---

## SHAP vs LIME

| Method | Strength |
|--------|----------|
| SHAP | Strong theoretical foundation, consistent feature attribution |
| LIME | Fast local explanations, intuitive segmentation-based insights |

Using both provides complementary understanding.

---

## Future Improvements

Possible extensions include:

- Explainability for custom-trained CNNs  
- Object detection explanations (YOLO / Faster R-CNN)  
- Medical image diagnosis interpretability  
- Video-based XAI analysis  
- Real-time webcam explanations  
- Bias and fairness auditing  

---

## Final Note

This project reflects my interest in combining:

- Deep Learning  
- Computer Vision  
- Trustworthy AI  
- Model Interpretability  
- Human-Centered Machine Learning  

to build systems that are not only accurate, but also understandable.

**Author: Endri Dibra**
