# Lesson 09: Dataset Preparation and Loading

## Learning Objectives
- Explain the purpose of train, validation, and test splits and describe the risks of omitting any split
- Load and preprocess the CIFAR-10 dataset using Keras built-in utilities
- Build an efficient data pipeline using `tf.data` with batching, shuffling, and prefetching
- Apply image augmentation using `ImageDataGenerator` and `tf.data` transforms
- Detect and visualize class imbalance in a dataset
- Describe the steps in a complete image preprocessing pipeline including resizing and normalization

---

## Detailed Explanation

### The Train / Validation / Test Split — and Why It Matters

When you train a machine learning model, you need to answer two separate questions:
1. *Is the model learning from the training data?*
2. *Will the model generalize to data it has never seen?*

These two questions require **three separate, non-overlapping portions of your data**:

**Training set (typically 70–80%):** The data the model actually learns from. Weights are updated based on loss calculated on this set. The model may eventually *memorize* this data (overfit), which is why you can't use it to judge generalization.

**Validation set (typically 10–15%):** Data the model never trains on, used to monitor generalization *during* training. You use this to make decisions like "stop training now" or "try a different architecture." Critically, if you use the validation set to make many architecture decisions, you implicitly optimize for it — which is called "validation set overfitting."

**Test set (typically 10–15%):** Data the model has absolutely never seen, touched only *once* at the very end to report final performance. This gives you the honest, unbiased estimate of real-world performance. Never use the test set during development.

```
All Available Data
├── Training Set   (70%) ← model updates weights here
├── Validation Set (15%) ← we monitor here; used to tune hyperparameters
└── Test Set       (15%) ← touched only once for final evaluation
```

**A common mistake:** Using the test set to choose between multiple models, then reporting the best result. This is a form of data leakage — the "test" result is no longer unbiased.

---

### Data Loading Strategies

**From NumPy Arrays (simplest):**
When data fits in memory (like MNIST or CIFAR-10), loading everything into NumPy arrays is the simplest approach:

```python
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
```

This loads all 60,000 CIFAR-10 images into RAM. Fine for small datasets, impractical for millions of high-resolution images.

**From Directories (for large datasets):**
Real-world datasets are stored as files on disk:
```
data/
  train/
    cats/  ← class name becomes the label
      img001.jpg
      img002.jpg
    dogs/
      img001.jpg
  val/
    cats/
    dogs/
```

Keras's `image_dataset_from_directory()` reads this structure, infers labels from folder names, and creates a `tf.data.Dataset` that loads images lazily (on demand).

**tf.data Pipeline (for production):**
The `tf.data` API creates efficient, parallelized data pipelines:
```python
dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.batch(32)
dataset = dataset.prefetch(tf.data.AUTOTUNE)
```

`prefetch(AUTOTUNE)` is particularly important — it tells TensorFlow to prepare the *next* batch in the background while the GPU processes the *current* batch, eliminating I/O bottlenecks.

---

### Image Preprocessing Steps

**1. Resizing:**
Neural networks require fixed-size inputs. Real-world images come in all sizes. You must resize all images to the same dimensions before feeding them to the network:
- Resize to the model's expected input (e.g., 32×32, 224×224)
- Maintain aspect ratio when possible (pad or crop instead of stretch)

**2. Normalization:**
Raw pixel values range from 0 to 255. Neural networks train much more efficiently with values in a smaller range. Two common approaches:
- **Min-max normalization:** divide by 255 → values in [0, 1]
- **Standardization:** subtract mean, divide by std → zero mean, unit variance

```python
x_train = x_train.astype('float32') / 255.0  # simple normalization
```

Without normalization, gradients from later layers would be enormous compared to early-layer gradients, causing unstable training.

**3. Data Type:**
Keras expects `float32`. Images loaded as `uint8` must be cast:
```python
x_train = x_train.astype('float32')
```

---

### Data Augmentation

Training images are often limited. A model trained on only a few thousand images will overfit — memorizing specific examples rather than learning general features. **Data augmentation** artificially expands the training set by applying random transformations to training images during each epoch:

```
Original image → randomly transformed → model trains on transformed version
```

Common augmentations for image classification:
| Augmentation | Effect | Intuition |
|---|---|---|
| Random horizontal flip | Mirror image | Objects look the same mirrored |
| Random rotation (±15°) | Slight tilt | Objects appear at various angles |
| Random zoom (0.8–1.2×) | Scale variation | Objects appear at various distances |
| Random brightness | Lighting variation | Same scene in different lighting |
| Random crop | Partial view | Model must recognize partial objects |
| Random width/height shift | Position variation | Object doesn't always center |

**Critical rule:** Apply augmentation *only to the training set*. Validation and test sets should be processed consistently without random transformations. Augmenting validation/test sets would make evaluation unreliable.

---

### ImageDataGenerator

Keras's `ImageDataGenerator` is a classic tool that applies augmentation on-the-fly during training, without storing augmented images to disk:

```python
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)
train_generator = datagen.flow(x_train, y_train, batch_size=32)
```

The generator yields batches of randomly augmented images. Each epoch sees a *different* random transformation of the same image, effectively creating an infinite supply of slightly different training examples.

**Modern alternative:** TensorFlow's `tf.keras.layers.RandomFlip`, `RandomRotation`, etc. allow augmentation to be defined as *layers inside the model*, which is cleaner and GPU-accelerated.

---

### Batching and Shuffling

**Batching:** Instead of updating weights after every single image (pure SGD) or after all images (batch gradient descent), we typically use **mini-batches** of 32–256 images:
```
batch_size = 32  ← most common choice
steps_per_epoch = len(x_train) / batch_size = 50000 / 32 ≈ 1562 steps
```

**Why batch?** Single-sample updates are noisy and slow; full-dataset updates require too much memory and miss the regularization benefit of noisy gradient estimates.

**Shuffling:** Before creating batches, shuffle the training data. If your data is sorted by class (all cats first, then all dogs), unshuffled batches would contain only one class. The model would see wildly inconsistent gradients batch-to-batch. Shuffling ensures each batch is a random sample from all classes.

```python
# Shuffle before each epoch (important!)
dataset = dataset.shuffle(buffer_size=len(x_train))
dataset = dataset.batch(32)
```

**Shuffle buffer size:** With `shuffle(10000)`, TensorFlow loads the first 10,000 elements into a buffer and randomly samples from that buffer. Larger buffer = better shuffling quality but more memory.

---

### Class Imbalance

Class imbalance occurs when some classes have far more training examples than others. For example, in a medical imaging dataset, 95% of samples might be "healthy" and only 5% "disease." A model that always predicts "healthy" would achieve 95% accuracy without learning anything useful.

**Detection:** Count samples per class and visualize with a bar chart:
```python
import numpy as np
unique, counts = np.unique(y_train, return_counts=True)
```

**Solutions:**
- **Oversample minority classes** — duplicate or augment rare class examples
- **Undersample majority classes** — remove some majority examples
- **Class weights** — tell Keras to penalize errors on minority classes more heavily
- **Synthetic data** — generate new examples using SMOTE or GANs

CIFAR-10 is perfectly balanced (5,000 training images per class), so class imbalance is not an issue there. But real-world datasets almost always exhibit some imbalance.

---

### Organizing a Complete Data Pipeline

A complete, production-quality data pipeline looks like this:

```
Raw files on disk
       ↓
Load & decode (image bytes → pixel arrays)
       ↓
Resize to target dimensions
       ↓
Cast to float32
       ↓
Normalize (divide by 255 or standardize)
       ↓
[Training only] → Augment (flip, rotate, zoom)
       ↓
Batch
       ↓
Shuffle (training) / no shuffle (val/test)
       ↓
Prefetch (overlap I/O with GPU computation)
       ↓
Feed to model.fit()
```

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Train Split | Subset of data used to update model weights | The model learns directly from this data |
| Validation Split | Subset not used for training; monitors generalization during development | Guides hyperparameter tuning without contaminating the test set |
| Test Split | Subset used only once for final evaluation | Gives unbiased estimate of real-world performance |
| Normalization | Scaling pixel values to [0,1] or zero-mean/unit-variance | Stabilizes and speeds up gradient descent |
| Data Augmentation | Randomly transforming training images to create variety | Reduces overfitting; artificially expands training set |
| ImageDataGenerator | Keras utility that applies augmentation on-the-fly during training | Classic tool for image augmentation without disk storage |
| tf.data Pipeline | TensorFlow's efficient, parallelized data loading API | Eliminates I/O bottlenecks; scales to large datasets |
| Batching | Grouping multiple samples into a single forward/backward pass | Balances computational efficiency and gradient estimate quality |
| Shuffling | Randomizing data order before batching | Ensures representative batches and stable training |
| Class Imbalance | When some classes have far fewer examples than others | Causes biased models; requires resampling or class weighting |
| Prefetching | Preparing next batch while GPU processes current batch | Eliminates CPU/GPU idle time; significant speedup |

---

## Code Reference

See [`code/lesson_09.py`](../code/lesson_09.py) for fully runnable demonstrations loading CIFAR-10, creating data splits, building tf.data pipelines, applying augmentation, and visualizing class distributions.

---

## Activities

1. **Manual Split Implementation:** Load CIFAR-10 using Keras. Without using `train_test_split`, manually create a validation set by taking the last 5,000 samples from the training set. Verify that your train, validation, and test sets have the expected sizes and non-overlapping indices.

2. **Normalization Comparison:** Train a small CNN on CIFAR-10 twice — once with raw uint8 pixel values (0–255) and once with normalized float32 values (0–1.0). Compare the loss curves. What problem do you observe with unnormalized training? Why does this happen?

3. **Augmentation Visualization:** Using `ImageDataGenerator`, apply at least 5 different augmentation types to a single CIFAR-10 image. Generate 16 augmented versions and display them in a 4×4 grid. What range of variation does each augmentation type create?

4. **tf.data Pipeline Construction:** Build a complete `tf.data` pipeline for CIFAR-10 that includes: loading from numpy arrays, casting, normalizing, shuffling, batching (batch size 64), and prefetching. Time how long it takes to iterate through the entire training set for 3 epochs. Compare to using the numpy arrays directly.

5. **Class Imbalance Simulation:** Take the CIFAR-10 training set and artificially create imbalance by keeping only 500 samples from class 0 (airplane) but all 5,000 samples from all other classes. Train a CNN on this imbalanced set. What per-class accuracy do you observe? Then apply class weights to compensate. Does it improve class 0 accuracy?

---

## Review Questions

1. Why do we need three separate data splits (train/validation/test) rather than just train and test? What specific problem does the validation set solve?

2. Explain the concept of "data leakage." Give two concrete examples of how data leakage can corrupt an evaluation result.

3. Why is shuffling the training data before batching important? What problem occurs if you train on sequential, unshuffled batches?

4. What is the purpose of `prefetch(tf.data.AUTOTUNE)` in a tf.data pipeline? Draw a timeline diagram showing the difference between a pipeline with and without prefetching.

5. Describe three augmentation techniques appropriate for image classification. For each, explain the real-world variation it simulates and identify any situation where that augmentation would be *inappropriate* (e.g., augmenting medical images where orientation matters).

---

## Further Reading

- **TensorFlow Data Pipeline Performance Guide** — https://www.tensorflow.org/guide/data_performance — Official guide with benchmarks and best practices
- **"A Survey on Image Data Augmentation for Deep Learning"** (Shorten & Khoshgoftaar, 2019) — Comprehensive review of augmentation techniques
- **Keras ImageDataGenerator documentation** — https://keras.io/api/preprocessing/image/ — Complete API reference
- **"Practical Machine Learning with Scikit-Learn and TensorFlow"** (Géron) — Chapter on data loading and preprocessing pipelines
- **Albumentations library** — https://albumentations.ai/ — Advanced augmentation library with 70+ techniques, widely used in competitions
