# 🧠 Image Classification Using CNNs — CS Elective 102

![Lessons](https://img.shields.io/badge/Lessons-20-blue) ![Python](https://img.shields.io/badge/Python-3.9%2B-yellow) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange) ![License](https://img.shields.io/badge/License-MIT-green)

---

## 📖 Course Description

Convolutional Neural Networks (CNNs) are the backbone of modern computer vision. Unlike traditional machine learning models that require hand-crafted features, CNNs automatically learn spatial hierarchies of features directly from raw pixel data — making them uniquely powerful for tasks like image classification, object detection, and face recognition. This course walks you through every layer of that process, from the math of convolutions to deploying a trained model in production.

Over 20 structured lessons, you will build a deep understanding of how CNNs work and why they work so well on images. You'll start from the fundamentals — what a digital image really is, how neural networks learn — and progressively advance to transfer learning, model visualization with Grad-CAM, and a full capstone project where you design, train, evaluate, and deploy your own image classifier end-to-end.

This course is designed for university-level CS students who know Python but have no prior machine learning experience. Every concept is explained intuitively before being formalized mathematically, and every lesson is accompanied by fully working Python code using TensorFlow/Keras. By the end, you will have practical, portfolio-ready experience building real CNN-based classifiers.

---

## 📋 Prerequisites

| Requirement | Details |
|---|---|
| Python basics | Variables, loops, functions, lists — no OOP required |
| Basic math | Algebra and an intuition for what a function is |
| No ML experience | Everything is taught from scratch |

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/rbpelegrinojr/LessonCSElective102.git
cd LessonCSElective102
```

### 2. (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run a lesson's code file

```bash
python code/lesson_01.py
```

### 5. Generate the full course PDF

```bash
python generate_pdf.py
```

This will create `CNN_Image_Classification_Course.pdf` in the root directory containing all 20 lessons with their lesson notes and Python code.

---

## 📚 Table of Contents

| # | Lesson Title |
|---|---|
| 01 | What Is Image Classification? |
| 02 | Understanding Digital Images & Pixels |
| 03 | Introduction to Neural Networks |
| 04 | From Dense Networks to CNNs |
| 05 | The Convolution Operation Deep Dive |
| 06 | Activation Functions |
| 07 | Pooling Layers |
| 08 | Building Your First Complete CNN |
| 09 | Dataset Preparation & Loading |
| 10 | Training a CNN |
| 11 | Loss Functions & Optimizers |
| 12 | Evaluating Your Model — Metrics & Confusion Matrix |
| 13 | Overfitting & Regularization Techniques |
| 14 | Data Augmentation |
| 15 | Batch Normalization & Dropout |
| 16 | Transfer Learning |
| 17 | Fine-Tuning Pretrained Models |
| 18 | Model Visualization & Interpretability (Grad-CAM) |
| 19 | Deploying Your CNN Model |
| 20 | Capstone Project — End-to-End Image Classifier |

---

## 🗂️ Folder Structure

```
LessonCSElective102/
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── generate_pdf.py                  # Script to compile all lessons into one PDF
├── CNN_Image_Classification_Course.pdf  # Generated output (after running generate_pdf.py)
├── lessons/
│   ├── lesson_01.md                 # Lesson notes in Markdown
│   ├── lesson_02.md
│   ├── ...
│   └── lesson_20.md
└── code/
    ├── lesson_01.py                 # Working Python code for each lesson
    ├── lesson_02.py
    ├── ...
    └── lesson_20.py
```

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **TensorFlow / Keras 2.10+** — model building and training
- **NumPy** — numerical operations
- **Matplotlib / Seaborn** — visualization
- **scikit-learn** — metrics and utilities
- **Pillow / OpenCV** — image handling
- **fpdf2** — PDF generation

---

*Course authored by rbpelegrinojr*
