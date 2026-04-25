# 🧠 Image Classification Using CNNs — CS Elective 102

> A comprehensive 20-lesson university-level course on image classification using Convolutional Neural Networks (CNNs) with TensorFlow/Keras. This course takes you from absolute beginner to building, training, evaluating, and deploying real image classifiers.

---

## 📋 Prerequisites

- Basic Python programming (variables, loops, functions)
- High-school level mathematics (algebra, basic calculus concepts helpful but not required)
- No prior machine learning or deep learning experience needed

---

## 🚀 Quick Setup

```bash
# 1. Clone this repository
git clone https://github.com/rbpelegrinojr/LessonCSElective102.git
cd LessonCSElective102

# 2. (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Run any lesson code
python code/lesson_01.py

# 5. Generate the full course PDF
python generate_pdf.py
# → Output: CNN_Image_Classification_Course.pdf
```

---

## 📄 Generating the Course PDF

The included `generate_pdf.py` script compiles all 20 lessons (markdown content + Python code) into a single, well-formatted PDF:

```bash
pip install fpdf2   # if not already installed
python generate_pdf.py
```

Output: **`CNN_Image_Classification_Course.pdf`** — a complete, printable course document with:
- Cover page
- Table of contents
- All 20 lesson explanations
- All 20 Python code listings
- Page numbers

---

## 📚 Course Curriculum — All 20 Lessons

| # | Lesson | Key Topics |
|---|--------|-----------|
| 01 | [What Is Image Classification?](lessons/lesson_01.md) | Pipeline, applications, history, computer vision overview |
| 02 | [Understanding Digital Images and Pixels](lessons/lesson_02.md) | RGB/grayscale, numpy arrays, normalization, color spaces |
| 03 | [Introduction to Neural Networks](lessons/lesson_03.md) | Perceptrons, weights, biases, forward pass, training loop |
| 04 | [From Dense Networks to CNNs](lessons/lesson_04.md) | Parameter explosion, spatial invariance, parameter sharing |
| 05 | [The Convolution Operation Deep Dive](lessons/lesson_05.md) | Kernels, stride, padding, feature maps, filter examples |
| 06 | [Activation Functions](lessons/lesson_06.md) | ReLU, sigmoid, tanh, softmax, vanishing gradients |
| 07 | [Pooling Layers](lessons/lesson_07.md) | Max pooling, average pooling, global average pooling |
| 08 | [Building Your First Complete CNN](lessons/lesson_08.md) | Full architecture, Keras Sequential API, LeNet-5 |
| 09 | [Dataset Preparation and Loading](lessons/lesson_09.md) | Train/val/test splits, tf.data pipeline, ImageDataGenerator |
| 10 | [Training a CNN](lessons/lesson_10.md) | Training loop, callbacks, learning curves, convergence |
| 11 | [Loss Functions and Optimizers](lessons/lesson_11.md) | Cross-entropy, MSE, SGD, Adam, learning rate scheduling |
| 12 | [Evaluating Your Model](lessons/lesson_12.md) | Precision, recall, F1, confusion matrix, ROC/AUC |
| 13 | [Overfitting and Regularization](lessons/lesson_13.md) | L1/L2, dropout, early stopping, bias-variance tradeoff |
| 14 | [Data Augmentation](lessons/lesson_14.md) | Geometric/color augmentation, MixUp, CutOut, TTA |
| 15 | [Batch Normalization and Dropout](lessons/lesson_15.md) | Internal covariate shift, BatchNorm, Dropout, interaction |
| 16 | [Transfer Learning](lessons/lesson_16.md) | ImageNet, VGG16, MobileNetV2, feature extraction |
| 17 | [Fine-Tuning Pretrained Models](lessons/lesson_17.md) | Gradual unfreezing, catastrophic forgetting, discriminative LRs |
| 18 | [Model Visualization with Grad-CAM](lessons/lesson_18.md) | Activation maps, Grad-CAM algorithm, interpretability |
| 19 | [Deploying Your CNN Model](lessons/lesson_19.md) | SavedModel, TFLite, quantization, Flask API, production |
| 20 | [Capstone Project](lessons/lesson_20.md) | End-to-end pipeline, EDA, error analysis, model card |

---

## 📁 Repository Structure

```
LessonCSElective102/
│
├── README.md                   ← This file — course overview
├── requirements.txt            ← All Python dependencies
├── generate_pdf.py             ← Compiles everything into one PDF
│
├── lessons/
│   ├── lesson_01.md            ← What Is Image Classification?
│   ├── lesson_02.md            ← Understanding Digital Images and Pixels
│   ├── lesson_03.md            ← Introduction to Neural Networks
│   ├── lesson_04.md            ← From Dense Networks to CNNs
│   ├── lesson_05.md            ← The Convolution Operation Deep Dive
│   ├── lesson_06.md            ← Activation Functions
│   ├── lesson_07.md            ← Pooling Layers
│   ├── lesson_08.md            ← Building Your First Complete CNN
│   ├── lesson_09.md            ← Dataset Preparation and Loading
│   ├── lesson_10.md            ← Training a CNN
│   ├── lesson_11.md            ← Loss Functions and Optimizers
│   ├── lesson_12.md            ← Evaluating Your Model
│   ├── lesson_13.md            ← Overfitting and Regularization
│   ├── lesson_14.md            ← Data Augmentation
│   ├── lesson_15.md            ← Batch Normalization and Dropout
│   ├── lesson_16.md            ← Transfer Learning
│   ├── lesson_17.md            ← Fine-Tuning Pretrained Models
│   ├── lesson_18.md            ← Model Visualization with Grad-CAM
│   ├── lesson_19.md            ← Deploying Your CNN Model
│   └── lesson_20.md            ← Capstone Project
│
└── code/
    ├── lesson_01.py            ← Image classification demo with MNIST
    ├── lesson_02.py            ← Pixel manipulation and color spaces
    ├── lesson_03.py            ← Neural network basics
    ├── lesson_04.py            ← Dense vs CNN comparison
    ├── lesson_05.py            ← Convolution operation demo
    ├── lesson_06.py            ← Activation function visualizations
    ├── lesson_07.py            ← Pooling layer demo
    ├── lesson_08.py            ← First complete CNN on MNIST
    ├── lesson_09.py            ← Dataset preparation with CIFAR-10
    ├── lesson_10.py            ← Full training with callbacks
    ├── lesson_11.py            ← Loss functions and optimizers
    ├── lesson_12.py            ← Metrics and confusion matrix
    ├── lesson_13.py            ← Overfitting and regularization
    ├── lesson_14.py            ← Data augmentation techniques
    ├── lesson_15.py            ← BatchNorm and Dropout
    ├── lesson_16.py            ← Transfer learning with MobileNetV2
    ├── lesson_17.py            ← Fine-tuning demonstration
    ├── lesson_18.py            ← Grad-CAM implementation
    ├── lesson_19.py            ← Model deployment
    └── lesson_20.py            ← End-to-end capstone project
```

---

## 🛠️ Dependencies

All dependencies are listed in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `tensorflow>=2.10.0` | Deep learning framework (CNNs, training) |
| `numpy>=1.23.0` | Numerical arrays and math |
| `matplotlib>=3.5.0` | Plotting and visualization |
| `scikit-learn>=1.1.0` | Metrics, confusion matrix, data splitting |
| `pillow>=9.0.0` | Image loading and manipulation |
| `seaborn>=0.11.0` | Statistical visualizations |
| `scipy>=1.9.0` | Scientific computing utilities |
| `opencv-python>=4.6.0` | Computer vision operations |
| `fpdf2>=2.7.0` | PDF generation |

---

## 🎓 Learning Path

```
Beginner                    Intermediate                  Advanced
   │                             │                            │
Lessons 1–4              Lessons 5–12                Lessons 13–20
What is AI?              How CNNs work               Optimization & Deployment
Pixels & images          First CNN build             Transfer learning
Neural nets intro        Training & evaluation       Grad-CAM, TFLite, API
```

---

## 📝 License

This course is created for educational purposes as part of CS Elective 102.

---

*Happy Learning! 🚀 Start with [Lesson 1](lessons/lesson_01.md)*
