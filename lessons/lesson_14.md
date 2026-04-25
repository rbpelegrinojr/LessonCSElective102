# Lesson 14: Data Augmentation

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what data augmentation is and why it improves generalization
- Apply common augmentation techniques (flip, rotation, crop, color jitter) using Keras and PIL
- Use `ImageDataGenerator` and Keras preprocessing layers for augmentation pipelines
- Distinguish between online and offline augmentation
- Implement test-time augmentation (TTA) and understand its benefits

---

## Detailed Explanation

### What Is Data Augmentation?

**Data augmentation** is the practice of artificially increasing the diversity of your training dataset by applying random transformations to existing training examples. Instead of collecting more labeled data — which is expensive and time-consuming — you generate new, plausible variations of the data you already have.

The core idea: if flipping a cat image horizontally still looks like a cat, the model should predict "cat" for the flipped version too. By training on both the original and the augmented versions, the model learns that the label is invariant to that transformation.

Data augmentation is one of the most cost-effective regularization techniques available. It reduces overfitting without any architectural changes to the model, simply by making it harder to memorize specific training examples.

### Why It Works: Generalization Through Invariance

The reason data augmentation improves generalization is that it explicitly teaches the model which transformations should **not** change the prediction:

- A rotated dog is still a dog.
- A brighter image of a car is still a car.
- A cropped view of a cat's face is still a cat.

Real-world test images will naturally have these variations (different lighting, camera angles, zoom levels), so training on augmented data prepares the model for the variation it will encounter at deployment time. Augmentation acts as a form of **domain randomization** — making the training distribution broader so the test distribution is more likely to fall within it.

### Common Augmentation Techniques

#### Horizontal and Vertical Flip

One of the simplest and most effective techniques. For many natural image datasets (animals, vehicles, indoor scenes), horizontal flips are valid augmentations. **Vertical flips** must be used carefully — they are appropriate for aerial/satellite imagery but inappropriate for most everyday photos where gravity defines orientation.

#### Random Rotation

Rotating an image by a small random angle (e.g., ±15°) teaches rotation invariance. The range matters: rotating a handwritten letter by ±10° is fine, but ±45° might make it ambiguous. Extreme rotations (e.g., 180°) can completely change the meaning of a digit (e.g., turning a 9 into a 6).

The `fill_mode` parameter handles pixels at the borders after rotation: `nearest` fills with edge pixels, `reflect` mirrors the image, `constant` fills with a fixed value (often black).

#### Random Crop and Resize

Take a random rectangular crop of the image and resize it back to the original resolution. This achieves two things simultaneously: it shifts the subject's position (translation invariance) and changes the effective zoom level. **Random resized crop** is the dominant augmentation used in ImageNet training (e.g., in ResNet, EfficientNet papers).

#### Color Jitter: Brightness, Contrast, Saturation, Hue

Color jitter randomly perturbs the photometric properties of an image within a specified range:
- **Brightness:** Make the image lighter or darker
- **Contrast:** Increase or decrease the difference between light and dark areas
- **Saturation:** Make colors more vivid or more grey
- **Hue:** Slightly shift the color tone

These transforms simulate different lighting conditions and camera sensors. A model trained with color jitter is more robust to deployment in different lighting environments.

#### Zoom In/Out

Randomly zoom into the image (making the subject larger in the frame) or zoom out (making it smaller, with padding at the borders). This teaches scale invariance — the model should recognize objects regardless of how large or small they appear.

#### Shear

Shear transformation slants the image along one axis, like pressing one corner while keeping the opposite corner fixed. This simulates viewing the subject from a slightly tilted perspective.

#### Cutout / Random Erasing

**Cutout** (also called Random Erasing) randomly masks out a rectangular patch of the image by setting pixels to zero (or random noise). This forces the model to classify based on partial information, which improves robustness to occlusion. It is particularly effective for fine-grained classification tasks.

### When NOT to Augment: The MNIST 6 vs 9 Problem

Augmentation must be **label-preserving**. The critical question is always: "Does this transformation preserve the semantic meaning of the label?"

A notable failure case: if you apply 180° rotation augmentation to MNIST digit images, the digit 6 becomes visually identical to a 9, and vice versa. This introduces contradictory training examples — the same image appearing with two different labels — which can degrade model performance.

Other cases where standard augmentation is problematic:
- **Medical imaging:** Flipping left/right structures in cardiac or lung images can make them anatomically incorrect
- **Text in images:** Flipping or rotating text renders it unreadable/meaningless
- **Directional scenes:** An image where "up" carries semantic meaning should not be vertically flipped

Always validate that your chosen augmentations make domain sense before adding them to your pipeline.

### Online vs Offline Augmentation

**Offline augmentation:** Generate augmented images before training and save them to disk. The dataset is fixed and larger. Advantage: fast data loading during training. Disadvantage: the model sees the exact same augmented images every epoch, potentially still memorizing them. Requires significant extra disk space.

**Online augmentation:** Apply random transformations on-the-fly during training, typically per mini-batch. Every epoch, each training image is transformed differently, yielding effectively unlimited unique examples. This is the standard modern approach and provides better regularization. The tradeoff is a slight increase in CPU/data preprocessing cost.

### Keras ImageDataGenerator

The `ImageDataGenerator` class (from `tf.keras.preprocessing.image`) is the traditional Keras API for augmentation. It supports:
- `horizontal_flip`, `vertical_flip`
- `rotation_range` (in degrees)
- `width_shift_range`, `height_shift_range` (translation as fraction of width/height)
- `zoom_range`
- `shear_range`
- `brightness_range`
- `fill_mode` (how to fill pixels after geometric transforms)

Use `.flow(X_train, y_train)` for in-memory arrays or `.flow_from_directory()` for loading from disk.

### Keras Augmentation Layers (Modern Approach)

TensorFlow 2.x introduced `tf.keras.layers` preprocessing layers that can be built directly into the model graph. This means augmentation runs on the GPU and is exported with the model:

```python
augment = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
```

These layers are **only active during training** — they become identity functions at inference time.

### How Augmentation Affects Training Time

Online augmentation increases data preprocessing time. Each image must be randomly transformed before being fed to the model. For complex augmentation pipelines (especially those with many operations), this can become a bottleneck. Strategies to mitigate:
- Use `tf.data` pipelines with `.prefetch(tf.data.AUTOTUNE)` to overlap preprocessing with GPU computation
- Use `num_parallel_calls=tf.data.AUTOTUNE` for parallel CPU preprocessing
- Run augmentation directly on GPU using Keras layers

Typically, expect 1.2–2× increase in epoch time with standard augmentation.

### Test-Time Augmentation (TTA)

**Test-Time Augmentation** applies multiple augmentations to a single test image, generates a prediction for each, then **averages the predictions** (for probability outputs) or takes the majority vote (for class labels).

```
Input image → [original, flip, rotate +5°, rotate -5°, crop1, crop2]
                   ↓           ↓       ↓           ↓      ↓     ↓
              [0.8 cat, 0.85 cat, 0.79 cat, 0.81 cat, 0.83, 0.80]
                             ↓
                    Average → 0.81 (cat)  ← more confident
```

TTA consistently improves accuracy by 0.5–2% on most benchmarks with no additional training cost. It is widely used in competitions (Kaggle) and production systems where prediction latency is acceptable.

### ASCII: One Image → Multiple Augmented Versions

```
Original:           Flipped:           Rotated:          Cropped:
+----------+        +----------+       +----------+      +------+
|   🐱    |        |    🐱   |       |  🐱      |      | 🐱  |
|          |   →    |          |   →   |     (15°)|  →   |      |
+----------+        +----------+       +----------+      +------+

Color Jitter:       Cutout:            Zoomed In:
+----------+        +----------+       +----------+
|   🐱    |        |   🐱    |       |          |
| (bright) |   →    | [■■■■]  |   →   |   🐱    |
+----------+        +----------+       +----------+
```

---

## Key Concepts Table

| Technique | What It Does | Label-Safe? |
|---|---|---|
| Horizontal Flip | Mirror left-right | Usually yes |
| Vertical Flip | Mirror top-bottom | Domain-specific |
| Rotation | Rotate by random angle | For small angles |
| Random Crop | Crop + resize patch | Yes |
| Color Jitter | Random brightness/contrast/saturation | Yes |
| Zoom | Scale in or out | Yes |
| Shear | Slant along axis | Yes |
| Cutout | Mask rectangular region | Yes |
| TTA | Average predictions over augmented views | N/A (inference only) |

---

## Code Reference

```python
# Modern Keras augmentation layers (runs on GPU, inactive at test time)
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(factor=0.1),
    tf.keras.layers.RandomZoom(height_factor=0.1),
    tf.keras.layers.RandomContrast(factor=0.1),
])

# Legacy ImageDataGenerator
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
)
datagen.fit(X_train)
```

---

## Activities

1. **Augmentation Gallery:** Take a single image from CIFAR-10. Apply each augmentation technique individually and display the results in a grid. Note which augmentations look realistic.

2. **Augmentation Impact Study:** Train two identical CNNs on CIFAR-10 — one with and one without augmentation. Plot validation accuracy curves. Quantify the improvement.

3. **MNIST Danger Zone:** Apply 180° rotation augmentation to MNIST and train a model. Compare accuracy to a model trained without rotation. Explain the result.

4. **TTA Implementation:** Implement TTA with 8 augmented versions of each test image. Compare accuracy with and without TTA on CIFAR-10.

---

## Review Questions

1. Why does data augmentation improve generalization even without changing the model architecture?
2. What is the "MNIST 6 vs 9 problem" and what principle does it illustrate?
3. Compare online and offline augmentation. Which provides stronger regularization and why?
4. How do Keras preprocessing layers differ from `ImageDataGenerator`?
5. What is test-time augmentation and how does it improve prediction accuracy?
6. Name two augmentation techniques that would be appropriate for satellite imagery but not for photos of handwritten text.
7. How does augmentation increase effective dataset size compared to offline augmentation?

---

## Further Reading

- Shorten, C., & Khoshgoftaar, T. M. (2019). *A survey on Image Data Augmentation for Deep Learning*. Journal of Big Data.
- Cubuk, E. D., et al. (2019). *AutoAugment: Learning Augmentation Strategies from Data*. CVPR.
- DeVries, T., & Taylor, G. W. (2017). *Improved Regularization of Convolutional Neural Networks with Cutout*. arXiv:1708.04552.
- TensorFlow Data Augmentation Tutorial: https://www.tensorflow.org/tutorials/images/data_augmentation
- Keras ImageDataGenerator: https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
