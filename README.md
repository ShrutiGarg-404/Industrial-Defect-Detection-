# 🔍 Industrial Defect Detection

An AI-powered defect detection system that identifies surface defects in industrial products using Computer Vision and Deep Learning techniques.

This project is being developed as part of my internship at CSIR-CSIO, Chandigarh and focuses on automating quality inspection using the MVTec Anomaly Detection (MVTec AD) dataset.

---

## 📌 Project Overview

Manual inspection of manufactured products can be time-consuming and prone to human error. The goal of this project is to build a deep learning-based system that can:

* Detect whether a product is defective or non-defective
* Learn visual defect patterns from industrial images
* Highlight defect regions using explainable AI techniques
* Provide an easy-to-use interface for testing new images

---

## 🧠 Technologies Used

* Python
* PyTorch
* OpenCV
* NumPy
* Matplotlib
* Streamlit
* Transfer Learning (ResNet18)

---

## 📊 Dataset

This project uses the **MVTec Anomaly Detection (MVTec AD)** dataset, a widely used benchmark for industrial defect detection.

Selected categories:

* Leather
* Tile
* Metal Nut

Dataset Link:
https://www.mvtec.com/company/research/datasets/mvtec-ad

---

## 🚀 Project Roadmap

### Phase 1

* Dataset exploration and preprocessing
* Binary classification (Defective vs Non-Defective)

### Phase 2

* Transfer learning using ResNet18
* Model evaluation and performance analysis

### Phase 3

* Defect localization using Grad-CAM
* Visualization of defect regions

### Phase 4

* Streamlit web application deployment

---

## 📂 Project Structure

```text
industrial-defect-detection/
│
├── data/
├── notebooks/
├── src/
├── models/
├── outputs/
├── app/
├── requirements.txt
└── README.md
```

---

## 📈 Evaluation Metrics

Model performance will be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

---

## 🎯 Future Enhancements

* Support for additional MVTec categories
* Real-time defect inspection
* Multi-class defect classification
* Deployment on edge devices

---

## 👩‍💻 Author

Shruti Garg

B.Tech CSE, CCET Chandigarh

Interests: Artificial Intelligence, Machine Learning, Computer Vision, and Web Development.


