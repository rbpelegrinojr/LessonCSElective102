# Lesson 14: Data Augmentation

## Learning Objectives
- Explain why data augmentation improves generalization by forcing models to learn invariant features
- Distinguish between geometric augmentations (rotation, flip, zoom, shift, shear) and color augmentations (brightness, contrast, hue)
- Implement standard augmentation pipelines using Keras preprocessing layers and ImageDataGenerator
- Describe advanced augmentation techniques including Cutout, MixUp, and CutMix
- Select appropriate augmentation strategies for different domains (medical imaging vs. natural images)
- Evaluate model performance with and without augmentation on CIFAR-10

---

## Detailed Explanation

### Why Augmentation Works: Teaching Invariance

Imagine training a cat classifier using only photos of cats sitting upright, facing the camera, in bright light. If you then show it a photo of a cat lying on its side in dim light, it may fail — even though it has "seen" thousands of cats. The problem is that the model learned specific features tied to the exact training conditions rather than invariant features that characterize "cat-ness" regardless of pose, lighting, or framing.

**Data augmentation** artificially creates new training examples by applying label-preserving transformations to existing images. If we train on images that have been rotated, flipped, cropped, and color-shifted, the model must learn features that are stable across these variations. The label "cat" must be correctly predicted whether the cat is rotated 30° or 90°, making the model robust to orientation. This is called **learning invariances**.

Think of it as deliberate practice for athletes: a basketball player who only practices free throws from the same spot will struggle when the game situation varies. A player who practices from different angles, with different pressures, under varied conditions, builds generalizable skill. Augmentation is deliberate practice variability for neural networks.

Augmentation also serves as an implicit regularizer: by making each epoch contain slightly different images, it reduces the chance of the model memorizing specific training examples, increasing the effective dataset size exponentially.

---

### Geometric Augmentations

These transformations change the spatial layout of the image:

**Horizontal Flip**: Mirror the image left-to-right. Valid for natural images (a flipped cat is still a cat). NOT valid for text recognition (a flipped letter 'd' becomes 'b') or certain medical images (left vs. right lung matters).

**Rotation**: Rotate the image by a random angle θ degrees. Small angles (±15°) are safe for most tasks. For satellite imagery or medical scans, ±180° may be appropriate. Watch for edge artifacts — Keras fills with nearest pixel or black by default.

**Zoom / Scale**: Randomly zoom in or out by a factor. Zoom in (crop) simulates the subject being closer; zoom out (pad) simulates it being farther away. Useful for scale invariance.

**Translation (Shift)**: Move the image horizontally and/or vertically by a fraction of its dimensions. Teaches the model that the object doesn't have to be centered. `width_shift_range=0.1` shifts by up to 10% of width.

**Shear**: Apply a shear transformation — slanting the image along an axis. Simulates viewing an object from a slight angle.

```
Original:    Flipped H:   Rotated:    Shifted:    Sheared:
┌──────┐     ┌──────┐     ┌──────┐    ┌──────┐    ┌──────┐
│ /\_/\│     │/\_/\ │     │ \_/\ │    │      │    │ /\_  │
│(='.'=)     │='.') │     │='.') │    │/\_/\ │    │='.'  │
│(")_(")     │_(")( │     │(")(")│    │='.') │    │(")_  │
└──────┘     └──────┘     └──────┘    └──────┘    └──────┘
```

**Vertical Flip**: Valid for satellite imagery, texture classification, and some medical images. Inappropriate for natural scene photos (an upside-down sky is unrealistic).

---

### Color Augmentations

These transformations change pixel intensity values:

**Brightness Adjustment**: Randomly increase or decrease the brightness of the image. Simulates different lighting conditions (e.g., outdoor vs. indoor, morning vs. evening).

**Contrast Adjustment**: Modify the difference between light and dark pixels. Low contrast (foggy scene) vs. high contrast (bright sunlight) are both realistic.

**Saturation Adjustment**: Change the intensity of colors. Desaturating toward grayscale simulates faded photos; increasing saturation simulates vivid conditions.

**Hue Shift**: Rotate colors around the color wheel. A red apple with a slight hue shift becomes orange-red — still identifiable as an apple. Hue shift of ±10° is realistic; ±180° would turn it blue (unrealistic for most tasks).

**Channel Shuffle / Channel Dropout**: Randomly reorder or drop color channels. Encourages the model not to rely on any single color channel exclusively.

---

### Advanced Augmentation Techniques

**Cutout (Random Erasing)**: Randomly mask a rectangular patch of the input image with a solid color (often the mean pixel value or a random color). This forces the model to recognize objects from partial views, improving robustness to occlusion.

```
Before Cutout:    After Cutout:
┌──────────┐      ┌──────────┐
│   🐱     │      │   🐱     │
│          │  →   │    ████  │
│          │      │    ████  │
└──────────┘      └──────────┘
```

**MixUp**: Blend two training images and their labels by a mixing factor λ ∈ [0, 1]:
```
x_mix = λ * x_i + (1 - λ) * x_j
y_mix = λ * y_i + (1 - λ) * y_j
```
The model must predict a mixture of labels, which acts as a very strong regularizer and improves calibration of probability estimates.

**CutMix**: Cut a rectangular region from one image and paste it into another, mixing labels proportionally to the area of the cut region. Combines the spatial locality of Cutout with the label mixing of MixUp.

**RandAugment**: Automatically select N random augmentation operations from a library of 14 operations and apply them with magnitude M. Reduces the hyperparameter search of manual augmentation design.

---

### Augmentation Strategies by Domain

| Domain | Recommended Augmentations | Avoid |
|--------|--------------------------|-------|
| Natural Images (CIFAR, ImageNet) | Flip H, rotation ±30°, brightness, zoom | Vertical flip, large rotation |
| Medical Imaging (X-ray, MRI) | Rotation ±180°, zoom, brightness | Hue shift (clinical color matters) |
| Satellite/Aerial Imagery | All rotations, flip H & V, zoom | None of the geometric transforms are dangerous |
| Text/Document Recognition | Small rotation ±5°, brightness | Flip, large rotation, shear |
| Microscopy (cell images) | Full rotation, flip H & V, stain color normalization | — |

The golden rule: only apply augmentations that produce images the model might realistically encounter at test time. Augmentations that generate physically impossible images (e.g., upside-down text) can hurt performance.

---

### How Much Augmentation Is Too Much?

Augmentation strength is a hyperparameter. Too little augmentation provides limited benefit; too much augmentation makes training examples so different from test examples that the model underfits or trains on unrealistic images.

Signs that augmentation is too aggressive:
- Training loss increases while validation loss stays flat (training examples too hard)
- Model does worse with augmentation than without
- Augmented images look obviously unrealistic when visualized

Practical guideline: Always visualize augmented samples before training. If you would struggle to label them correctly, the augmentation is too aggressive.

---

### Test-Time Augmentation (TTA)

**TTA** applies augmentation at inference time and averages the predictions from multiple augmented versions of the same test image. For example, for a single test image, generate 10 augmented versions, run each through the model, and average the output probability vectors.

```
Test image → [original, flip, rotate+15°, rotate-15°, zoom 90%]
           → [pred1, pred2, pred3, pred4, pred5]
           → Average probabilities → final prediction
```

TTA typically improves accuracy by 0.5–2% at the cost of increased inference time. It is commonly used in competitive machine learning.

---

### Keras Augmentation Implementation

**ImageDataGenerator** (legacy API): Applies augmentations on-the-fly during training via the fit generator pattern. Runs on CPU in parallel with GPU training.

**Keras Preprocessing Layers** (modern API): Augmentation layers like `RandomFlip`, `RandomRotation`, `RandomZoom` are part of the model itself. They run on the GPU and can be toggled between training and inference mode. This approach is preferred for new code as it bundles augmentation with the model, making deployment simpler.

```python
augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])
```

---

### Common Misconceptions

1. **"More augmentation always helps"** — Not true. Aggressive augmentation can cause underfitting if training images no longer resemble the distribution.
2. **"Augmentation is a substitute for more data"** — Augmentation helps, but real diverse data is better. Augmentation creates correlated variations; new real data adds truly independent samples.
3. **"Augmentation should be applied to the test set"** — Standard test evaluation uses no augmentation (except for TTA, which is a deliberate choice). Evaluating on augmented test data would give misleading results.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Data Augmentation | Applying label-preserving transformations to training images | Expands effective dataset size; teaches invariance |
| Geometric Augmentation | Spatial transforms: flip, rotate, zoom, shift, shear | Teaches spatial invariance |
| Color Augmentation | Pixel value transforms: brightness, contrast, hue | Teaches lighting/color invariance |
| Cutout | Randomly masking a patch of the image | Improves robustness to partial occlusion |
| MixUp | Blending two images and their labels | Strong regularization; improves probability calibration |
| CutMix | Pasting a crop from one image into another with label mixing | Combines benefits of Cutout and MixUp |
| Test-Time Augmentation | Averaging predictions over multiple augmented test versions | Small but consistent accuracy improvement at inference |
| ImageDataGenerator | Legacy Keras API for on-the-fly augmentation | Widely used; CPU-based augmentation pipeline |
| Keras Preprocessing Layers | Modern GPU-based augmentation layers built into the model | Portable, efficient, GPU-accelerated augmentation |
| Label-Preserving Transform | Augmentation that does not change the true class | Essential constraint: augmentation must not change the answer |

---

## Code Reference

See the full runnable demo in **code/lesson_14.py**

---

## Activities

1. **Augmentation Gallery**: Apply the following augmentations to a single CIFAR-10 image and display a 3×4 grid: original, horizontal flip, vertical flip, rotation 30°, rotation 90°, zoom in, zoom out, width shift, height shift, brightness +, brightness -, shear. Label each subplot.

2. **Augmentation Impact Experiment**: Train a CNN on only 5,000 CIFAR-10 training images. Train once without augmentation and once with standard augmentation (flip, rotation, zoom, brightness). Compare test accuracy curves over 20 epochs.

3. **MixUp Implementation**: Implement MixUp augmentation from scratch. Sample λ from Beta(0.4, 0.4) for each batch. Visualize 5 MixUp examples showing the two source images and the blended result. Report if this improves CIFAR-10 accuracy.

4. **Domain-Appropriate Augmentation**: Find or generate a small dataset of handwritten digits (MNIST). Design an augmentation pipeline appropriate for digits (what you should and should NOT include). Justify each choice and measure accuracy compared to no augmentation.

5. **Test-Time Augmentation**: Using a trained CIFAR-10 classifier, implement TTA with 5 augmented versions per test image. Compare single-image accuracy vs. TTA accuracy on 1,000 test images. How much does TTA improve accuracy?

---

## Review Questions

1. Explain in plain terms why augmentation helps a CNN generalize better. What property is the network forced to learn?
2. You are building a classifier for chest X-rays to detect pneumonia. Which augmentations would you include and which would you absolutely exclude? Justify each decision.
3. How does MixUp differ from standard augmentation? Why is predicting a mixture of labels considered good regularization?
4. What is test-time augmentation (TTA), and when would you use it? What are its costs and benefits?
5. Compare `ImageDataGenerator` to Keras preprocessing layers. What are the advantages of using preprocessing layers?

---

## Further Reading

- "The Effectiveness of Data Augmentation in Image Classification using Deep Learning" — Perez & Wang, 2017
- "Improved Regularization of Convolutional Neural Networks with Cutout" — DeVries & Taylor, 2017
- "MixUp: Beyond Empirical Risk Minimization" — Zhang et al., ICLR 2018
- "RandAugment: Practical Automated Data Augmentation" — Cubuk et al., NeurIPS 2020
- Keras documentation: `tf.keras.layers.RandomFlip`, `RandomRotation`, `RandomZoom` and other preprocessing layers
