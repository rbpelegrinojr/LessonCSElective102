# Lesson 20: Capstone Project — End-to-End Image Classifier

## Learning Objectives

By the end of this lesson, you will be able to:

- Execute a complete machine learning pipeline from raw data to deployed model
- Apply all concepts from Lessons 1–19 in a single cohesive project
- Build a multi-class flower image classifier using transfer learning and fine-tuning
- Evaluate a model with confusion matrices, classification reports, and per-class accuracy
- Generate Grad-CAM visualizations for both correct and incorrect predictions
- Export the final model and build a production-ready inference function

---

## Detailed Explanation

### Overview of the Capstone Project

This final lesson brings together every concept introduced in the course. You will build an end-to-end image classification system from scratch: data loading, exploration, preprocessing, model training (two phases), evaluation, visualization, and deployment. The problem is a **5-class flower classification** task using the TensorFlow Flowers dataset (or CIFAR-10 as a fallback), a realistic and manageable benchmark that exercises all the skills developed throughout the course.

### Lessons 1–19: How They All Connect

Before diving into the project steps, here is a summary of the full course and how each lesson contributes to this capstone:

| Lesson | Topic | Used in Capstone |
|---|---|---|
| 1 | What is Machine Learning? | Conceptual foundation |
| 2 | Introduction to Neural Networks | Architecture understanding |
| 3 | Perceptrons and Activation Functions | Dense layer design |
| 4 | Forward and Backward Propagation | Training loop intuition |
| 5 | Introduction to CNNs | Core model architecture |
| 6 | Convolutional Layers | Backbone feature extraction |
| 7 | Pooling and Padding | Spatial reduction in backbone |
| 8 | Building CNNs in Keras | Model construction syntax |
| 9 | Training CNNs | Compile, fit, callbacks |
| 10 | Overfitting and Regularization | Dropout in head |
| 11 | Data Augmentation | Training pipeline |
| 12 | Batch Normalization | Backbone internals |
| 13 | Advanced Architectures | MobileNetV2 backbone |
| 14 | Hyperparameter Tuning | LR, batch size choices |
| 15 | Evaluation Metrics | Confusion matrix, F1 |
| 16 | Transfer Learning | Feature extraction phase |
| 17 | Fine-Tuning | Fine-tuning phase |
| 18 | Grad-CAM | Final visualization |
| 19 | Deployment | Model export and inference |
| **20** | **Capstone** | **All of the above** |

### Problem Statement

**Task**: Build a multi-class classifier for 5 flower species:
- 🌹 Roses
- 🌷 Tulips
- 🌻 Sunflowers
- 🌼 Daisies
- 🌾 Dandelions

**Dataset**: TensorFlow Flowers dataset (~3,670 images, ~700 per class). Download via `tensorflow_datasets` or directly from Google's storage. CIFAR-10 serves as a fallback (10 classes, 60,000 images, 32×32).

**Success criteria**: ≥ 85% validation accuracy, interpretable Grad-CAM visualizations, exportable inference function.

### Step 1: Data Collection and Organization

The TensorFlow Flowers dataset is organized as one folder per class. Using `tf.keras.utils.image_dataset_from_directory`, you can load it directly into a `tf.data.Dataset` pipeline with automatic label inference from folder names.

```
flowers/
  daisy/        (~633 images)
  dandelion/    (~898 images)
  roses/        (~641 images)
  sunflowers/   (~699 images)
  tulips/       (~799 images)
```

**EDA tasks**:
- Count images per class → check for class imbalance
- Compute image size distribution → choose resize target
- Visualize a random grid of sample images from each class

### Step 2: Exploratory Data Analysis (EDA)

Before training, always understand your data:
- **Class distribution**: Is it balanced? Use `Counter` or `value_counts`.
- **Image statistics**: Mean, std, min, max pixel values.
- **Sample visualization**: Plot a 5×5 grid of random samples per class.
- **Outlier detection**: Very small/large images, corrupted files, near-duplicate images.

EDA prevents surprises during training and informs preprocessing decisions (resize target, normalization strategy).

### Step 3: Data Pipeline — Augmentation and Normalization

A robust `tf.data` pipeline:

```python
AUTOTUNE = tf.data.AUTOTUNE

augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

def prepare(ds, augment=False):
    ds = ds.map(lambda x, y: (tf.image.resize(x, [96, 96]), y))
    if augment:
        ds = ds.map(lambda x, y: (augmentation(x, training=True), y))
    return ds.cache().shuffle(1000).prefetch(AUTOTUNE)
```

Apply augmentation only to the training set, not validation or test sets.

### Step 4: Model Selection — Transfer Learning with MobileNetV2

MobileNetV2 is the backbone of choice for this project because:
- Small (3.4M parameters) — fast to train and easy to experiment with
- High accuracy on ImageNet (71.8% top-1)
- Available in `tf.keras.applications` with pretrained ImageNet weights
- Designed for efficiency (depthwise separable convolutions)

Build the model with `include_top=False` and add a custom classification head:

```
MobileNetV2 Backbone (frozen)
         │
   GlobalAveragePooling2D
         │
   Dense(128, relu)
         │
   Dropout(0.3)
         │
   Dense(5, softmax)   ← 5 flower classes
```

### Step 5: Training — Two Phases

**Phase 1 (Feature Extraction, 5 epochs)**:
- Freeze all MobileNetV2 layers
- Train only the new head
- LR = 1e-3 (Adam default)
- Expected validation accuracy: ~75–80%

**Phase 2 (Fine-Tuning, 5 more epochs)**:
- Unfreeze the top 30 layers of MobileNetV2
- Recompile with LR = 1e-5
- Expected validation accuracy: ~85–92%

Track both phases in a combined history plot.

### Step 6: Evaluation

After training:
- **Confusion matrix**: Which classes are most confused with each other?
- **Classification report**: Precision, recall, F1-score per class
- **Per-class accuracy bar chart**: Visualize where the model performs best/worst
- **Top-k accuracy**: What fraction of true labels are in the top-2 predictions?

```
ASCII Diagram: Full ML Pipeline

Raw Data → EDA → Augmentation Pipeline
    │
    ▼
Feature Extraction Phase (5 epochs, frozen backbone)
    │
    ▼
Fine-Tuning Phase (5 epochs, top layers unfrozen)
    │
    ▼
Evaluation (confusion matrix, F1, per-class accuracy)
    │
    ▼
Visualization (Grad-CAM on correct + incorrect predictions)
    │
    ▼
Export (SavedModel + TFLite) → Inference Function
```

### Step 7: Grad-CAM Visualization

Apply Grad-CAM (from Lesson 18) to:
1. **5 correctly classified images** — verify the model focuses on flower petals/centers
2. **5 misclassified images** — understand what caused the error (background, occlusion, similar species)

This analysis often reveals actionable improvements: more augmentation, more data for confused classes, or fine-tuning deeper into the backbone.

### Step 8: Export and Deployment

Save the final model in SavedModel format. Convert to TFLite with dynamic quantization. Build a `predict_image(path)` function (from Lesson 19) that:
1. Loads and resizes the image
2. Applies `preprocess_input`
3. Runs inference
4. Returns the predicted class name and confidence score

### Project Extensions

After completing the core project, consider:
- **Try different backbones**: Replace MobileNetV2 with EfficientNetB0 or ResNet50. Does accuracy improve?
- **Custom dataset**: Collect your own images using Google Images or a smartphone. How well does the pipeline transfer?
- **Web application**: Wrap the inference function in a FastAPI endpoint and build a simple HTML frontend.
- **ONNX export**: Convert the SavedModel to ONNX and run inference with ONNX Runtime.
- **Kaggle competition**: Apply the same pipeline to a Kaggle image classification challenge.

### Reflection: What We've Learned

Over 20 lessons, you have progressed from understanding individual perceptrons to building, training, evaluating, interpreting, and deploying state-of-the-art convolutional neural networks. You now understand:

- Why CNNs work for images (local connectivity, weight sharing, translation invariance)
- How to build efficient data pipelines with augmentation
- When and how to apply transfer learning and fine-tuning
- How to diagnose model behavior using interpretability tools
- How to take a model from notebook to production

The journey from here leads to object detection (YOLO, SSD), image segmentation (U-Net, Mask R-CNN), video understanding, self-supervised learning, and vision transformers (ViT). The foundations you have built here apply to all of these.

---

## Key Concepts Table

| Concept | Lesson Introduced | Role in Capstone |
|---|---|---|
| CNN Architecture | 5–8 | Backbone structure |
| Data Augmentation | 11 | Training pipeline robustness |
| Transfer Learning | 16 | Phase 1 training |
| Fine-Tuning | 17 | Phase 2 training |
| Grad-CAM | 18 | Result visualization |
| Model Export | 19 | Deployment |
| Confusion Matrix | 15 | Model evaluation |
| tf.data Pipeline | 9 | Data loading efficiency |
| Dropout | 10 | Head regularization |
| GlobalAveragePooling2D | 16 | Head architecture |

---

## Activities

1. **Run the Full Pipeline:** Execute `code/lesson_20.py` end to end on the TensorFlow Flowers dataset. Record and print: final validation accuracy, total training time (Phase 1 + Phase 2), and model file size on disk.

2. **Backbone Swap:** Replace the MobileNetV2 backbone with `tf.keras.applications.EfficientNetB0`. Retrain the full two-phase pipeline on the same Flowers dataset. Compare and print: final validation accuracy, total parameter count (trainable vs. frozen), and average training time per epoch for both backbones side by side.
## Review Questions

1. What are the 8 main steps of the end-to-end ML pipeline in this capstone?
2. Why do we train in two phases (feature extraction then fine-tuning)?
3. What does the confusion matrix tell you that overall accuracy does not?
4. How would you handle a severely imbalanced dataset (e.g., 900 roses but only 50 tulips)?
5. What is the purpose of Grad-CAM in a production system, beyond academic curiosity?
6. How do you choose between saving as `.h5` vs. SavedModel for this project?
7. Name three ways you could extend this project after achieving 85% accuracy.
8. What concepts from this course would you need to study further to tackle object detection?

---

## Further Reading

- [TensorFlow Flowers Dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers)
- [End-to-End Deep Learning Pipeline – TF Tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [Practical Machine Learning for Computer Vision – O'Reilly](https://www.oreilly.com/library/view/practical-machine-learning/9781098102357/)
- [Fast.ai Practical Deep Learning Course](https://course.fast.ai/)
- [Papers with Code: Image Classification](https://paperswithcode.com/task/image-classification)
- [Dive into Deep Learning (free textbook)](https://d2l.ai/)
