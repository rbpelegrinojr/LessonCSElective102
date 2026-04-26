# Lesson 9: Dataset Preparation & Loading

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why data quality is the foundation of any successful ML project
- Implement proper train/validation/test splits with correct ratios
- Understand and prevent data leakage in ML pipelines
- Load image datasets using Keras utilities and tf.data.Dataset
- Preprocess images: resize, normalize, and convert data types
- Recognize class imbalance and apply basic mitigation strategies

---

## Detailed Explanation

### The Foundation: Data Quality

There is a guiding principle in machine learning that shapes every professional's workflow: **"Garbage in, garbage out."** No matter how sophisticated your model architecture is, no matter how carefully you tune hyperparameters, if your data is noisy, mislabeled, imbalanced, or improperly split, your model will fail to generalize. In practice, experienced ML practitioners spend 60–80% of their time on data preparation — not on modeling.

What constitutes good data? Several properties matter:
- **Sufficient quantity:** Deep learning models are data-hungry. CNNs often need thousands of examples per class to generalize well.
- **Representative coverage:** Training data must include the range of conditions the model will encounter at inference time (different lighting, angles, backgrounds).
- **Correct labels:** Even 5% label noise can measurably reduce accuracy.
- **Appropriate preprocessing:** Raw pixels need to be normalized to stable ranges for gradient-based optimization to work efficiently.

---

### Train / Validation / Test Split Strategy

Every dataset must be divided into at least three parts before any training begins:

**Training set:** The data the model learns from. Gradient updates are computed on this set.

**Validation set:** A held-out subset used to monitor training progress and tune hyperparameters (learning rate, architecture choices). The model never trains directly on this data, but you do use it to make decisions — which indirectly influences the model.

**Test set:** The final, untouched evaluation set. It is used exactly once, after all development is complete, to report the honest, unbiased performance of the model. If you look at test set performance during development and make changes, you have effectively used the test set for decision-making and your reported accuracy is optimistic.

**Common split ratios:**

| Dataset Size         | Train | Validation | Test |
|----------------------|-------|------------|------|
| Small (<5K)          | 70%   | 15%        | 15%  |
| Medium (5K–100K)     | 80%   | 10%        | 10%  |
| Large (>100K)        | 90%   | 5%         | 5%   |

For very large datasets, even 1% for validation/test may be hundreds of thousands of samples — more than enough for reliable evaluation.

---

### Data Leakage: The Silent Killer

Data leakage occurs when information from the test set (or future data) **influences the training process** in any way. It causes models to appear far more accurate during evaluation than they will be in production.

**Common forms of data leakage:**

1. **Preprocessing leakage:** Computing normalization statistics (mean, std) on the entire dataset including the test set, then normalizing everything. The correct approach is to compute statistics on the training set only, then apply them to validation and test sets.

2. **Temporal leakage:** In time-series data, using future data points to predict past ones. Always split by time, not randomly.

3. **Duplicate leakage:** If the same image appears in both train and test sets (common in web-scraped datasets), the model memorizes it rather than learning to generalize.

4. **Label leakage:** Using features that directly encode the label. For example, including a "diagnosis code" field when predicting a medical condition.

5. **Augmentation leakage:** Applying data augmentation before splitting, so augmented versions of training images appear in the test set.

**Prevention rules:**
- Always split first, preprocess second
- Track data provenance carefully (where each image came from)
- Deduplicate your dataset before splitting
- Never look at test set distributions or statistics during development

---

### Directory Structure for Image Datasets

The most common format for image classification datasets organizes images by class in subdirectories:

```
dataset/
├── train/
│   ├── cat/
│   │   ├── cat_001.jpg
│   │   ├── cat_002.jpg
│   │   └── ...
│   └── dog/
│       ├── dog_001.jpg
│       └── ...
├── validation/
│   ├── cat/
│   └── dog/
└── test/
    ├── cat/
    └── dog/
```

Keras's `ImageDataGenerator.flow_from_directory()` and `tf.keras.utils.image_dataset_from_directory()` both read this structure automatically, inferring class labels from folder names.

---

### Keras ImageDataGenerator

ImageDataGenerator is the traditional Keras API for loading images with optional on-the-fly augmentation:

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale=1./255)
train_gen = datagen.flow_from_directory(
    'dataset/train/',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)
```

It performs lazy loading — images are loaded from disk one batch at a time, which keeps memory usage low even for large datasets.

---

### tf.data.Dataset API

The modern, preferred approach is `tf.data.Dataset`. It provides a high-performance pipeline with support for prefetching (loading the next batch while the current batch is being processed), caching, shuffling, and parallelized processing.

```
Data Source → Map (preprocess) → Shuffle → Batch → Prefetch → Model
```

The key performance methods:
- `.cache()`: Stores preprocessed data in memory after the first pass
- `.shuffle(buffer_size)`: Randomly shuffles data to prevent batch ordering bias
- `.batch(batch_size)`: Groups samples into batches
- `.prefetch(tf.data.AUTOTUNE)`: Overlaps preprocessing and training for speed

---

### Preprocessing: Resize, Normalize, Convert dtype

**Resize:** All images in a batch must have the same dimensions. Resize to a fixed size (e.g., 224×224 for standard CNNs, 32×32 for CIFAR-scale experiments).

**Normalize:** Raw pixel values are integers in [0, 255]. Dividing by 255.0 converts them to float32 in [0.0, 1.0]. This is essential because:
- Gradient descent converges much faster when inputs are in a small, uniform range
- Neural networks assume inputs are roughly zero-centered and unit-variance for stable initialization

**Standardization (optional but often better):** Subtract the dataset mean and divide by the standard deviation per channel. ImageNet-pretrained models are typically standardized to mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].

**dtype conversion:** TensorFlow operations expect float32 by default. Converting from uint8 (raw images) to float32 must happen before normalization.

---

### Class Balancing: Handling Imbalanced Datasets

Real-world datasets are rarely balanced. Consider a medical imaging dataset with 95% healthy patients and 5% with disease. A model that predicts "healthy" for every sample achieves 95% accuracy but is completely useless.

**Detection:** Plot the class distribution histogram before training. If any class has fewer than 1/3 of the samples of the largest class, you have imbalance.

**Mitigation strategies:**
1. **Oversampling:** Duplicate or augment samples from underrepresented classes
2. **Undersampling:** Remove samples from overrepresented classes (loses data)
3. **Class weights:** Tell Keras to penalize errors on rare classes more heavily:
   ```python
   model.fit(X_train, y_train, class_weight={0: 1.0, 1: 19.0})
   ```
4. **Stratified splitting:** Ensure each split has the same class proportions as the full dataset. Use `sklearn.model_selection.train_test_split` with `stratify=y`.

---

### Stratified Splitting

When splitting, you want each split to reflect the original class distribution. Random splitting can accidentally put all examples of a rare class into the test set. Stratified splitting prevents this:

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)
```

---

### Loading Benchmark Datasets from Keras

Keras provides clean, pre-split versions of common benchmark datasets:

```python
# MNIST: 70K grayscale 28×28 images, 10 digit classes
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# CIFAR-10: 60K RGB 32×32 images, 10 object classes
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Fashion-MNIST: 70K grayscale 28×28 images, 10 clothing categories
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
```

---

### Batch Loading: Why We Cannot Load Everything at Once

A single 224×224 RGB image stored as float32 takes 224 × 224 × 3 × 4 bytes = ~600KB. ImageNet contains 1.2 million images: 1.2M × 600KB = ~720GB. This far exceeds available RAM. Batch loading solves this by keeping only one batch in memory at a time (e.g., 32 images ≈ 19MB).

---

### Dataset Split ASCII Diagram

```
┌──────────────────────────────────────────────────────┐
│                   Full Dataset (100%)                │
└───────────────────┬────────────────┬─────────────────┘
                    │                │
          ┌─────────▼──────┐  ┌──────▼──────────┐
          │  Training Set  │  │   Held-out Set  │
          │     (80%)      │  │     (20%)       │
          └────────────────┘  └──────┬──────────┘
                                     │
                        ┌────────────┴────────────┐
                        │                         │
               ┌────────▼───────┐       ┌─────────▼──────┐
               │ Validation Set │       │   Test Set      │
               │    (10%)       │       │    (10%)        │
               │ Used during    │       │ Used ONCE at    │
               │ training for   │       │ end for honest  │
               │ hyperparameter │       │ evaluation      │
               │ tuning         │       │                 │
               └────────────────┘       └────────────────┘
```

---

## Key Concepts Table

| Term                   | Definition                                                                  |
|------------------------|-----------------------------------------------------------------------------|
| Data leakage           | When test/future information contaminates the training process              |
| Stratified split       | Splitting while preserving class distribution proportions                   |
| Normalization          | Scaling pixel values from [0, 255] to [0.0, 1.0]                           |
| Prefetching            | Loading the next batch while the current batch trains                       |
| Class imbalance        | Unequal numbers of examples across classes                                  |
| ImageDataGenerator     | Keras utility for loading images from directory with optional augmentation  |
| tf.data.Dataset        | High-performance TensorFlow data pipeline API                               |
| Benchmark dataset      | Standard publicly available dataset used for comparing model performance    |

---

## Code Reference

```python
import tensorflow as tf
import numpy as np

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Normalize
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Create validation split
val_size = int(0.1 * len(X_train))
X_val, y_val = X_train[:val_size], y_train[:val_size]
X_train, y_train = X_train[val_size:], y_train[val_size:]

# Build tf.data pipeline
train_ds = (tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(10000)
            .batch(32)
            .prefetch(tf.data.AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((X_val, y_val))
          .batch(32)
          .prefetch(tf.data.AUTOTUNE))
```

---

## Activities

1. **Dataset Exploration:** Load CIFAR-10 with `tf.keras.datasets.cifar10.load_data()`. Print the shape, dtype, min, max, and mean of the raw training images. Use `np.unique(y_train, return_counts=True)` to confirm there are exactly 5 000 samples per class.

2. **Pipeline Benchmarking:** Build two `tf.data.Dataset` pipelines from the CIFAR-10 training data — one without `.prefetch()` and one with `.prefetch(tf.data.AUTOTUNE)`. Use `time.time()` to measure the wall-clock time to iterate through one full epoch of batches (batch size 64) for each and print the difference.
## Review Questions

1. What does "garbage in, garbage out" mean in the context of machine learning?
2. Why do we need three separate splits (train/val/test) rather than just two?
3. Describe two forms of data leakage and explain how to prevent each.
4. Why should you always split your data before computing normalization statistics?
5. What is the advantage of `tf.data.Dataset` over `ImageDataGenerator`?
6. What is stratified splitting and when is it essential?
7. Why do we normalize pixel values before training a CNN?
8. How does `.prefetch(tf.data.AUTOTUNE)` improve training speed?

---

## Further Reading

- [tf.data: Build TensorFlow Input Pipelines](https://www.tensorflow.org/guide/data)
- [Rules of Machine Learning — Google](https://developers.google.com/machine-learning/guides/rules-of-ml)
- [A Survey on Data Collection for Machine Learning](https://arxiv.org/abs/1811.03402)
- [The Importance of Stratification in ML](https://scikit-learn.org/stable/modules/cross_validation.html#stratified-k-fold)
- [Keras Preprocessing Documentation](https://keras.io/api/preprocessing/)
