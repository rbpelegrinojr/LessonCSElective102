# Lesson 08: Building Your First Complete CNN

## Learning Objectives
- Assemble a complete CNN architecture using the Conv→ReLU→Pool pattern
- Build multi-block CNN architectures with increasing filter depths
- Use both the Keras Sequential API and Functional API to define models
- Read and interpret `model.summary()` output to trace shape transformations
- Train a complete CNN on MNIST and evaluate its performance
- Recognize the architecture of LeNet-5 and explain its historical significance

---

## Detailed Explanation

### Putting It All Together: The CNN Blueprint

In previous lessons, you learned about individual CNN components in isolation: convolutional layers detect features, activation functions introduce non-linearity, and pooling layers reduce spatial dimensions. Now it's time to combine these pieces into a complete, working architecture.

A complete CNN follows a recognizable two-stage structure:

```
INPUT IMAGE
     ↓
┌─────────────────────────────┐
│   FEATURE EXTRACTION        │
│  [Conv → ReLU → Pool] × N   │  ← repeated N times
└─────────────────────────────┘
     ↓
┌─────────────────────────────┐
│   CLASSIFICATION            │
│  Flatten → Dense → Output   │  ← fully connected head
└─────────────────────────────┘
     ↓
OUTPUT (class probabilities)
```

The feature extraction stage learns *what* features are present in the image. The classification stage learns *how to combine* those features to make a prediction.

---

### The Conv → ReLU → Pool Pattern

This three-layer pattern is the fundamental building block of CNNs:

**Step 1 — Convolution:**
A `Conv2D` layer slides learned filters across the input, producing feature maps that highlight the presence of specific patterns (edges, textures, shapes). The filter weights are the learnable parameters.

**Step 2 — ReLU Activation:**
Applied element-wise to the convolutional output, ReLU zeroes out all negative values. This introduces non-linearity and sparse representation. Without this step, stacking multiple conv layers would be no more powerful than a single one.

**Step 3 — Max Pooling:**
Reduces spatial dimensions by 2× (with 2×2 pool and stride 2). This achieves three things simultaneously: smaller feature maps, larger effective receptive fields for the next layer, and partial translation invariance.

```
After block 1: 28×28×1  →  26×26×32  →  26×26×32  →  13×13×32
              (input)   (after conv)  (after relu)  (after pool)
```

**Why increase filter depth as we go deeper?**

Early layers detect simple features (horizontal edges, vertical edges, dots). These simple features combine to form textures, which combine to form shapes, which combine to form objects. As complexity increases, you need *more* channels to represent the growing vocabulary of features. It's standard practice to roughly double the number of filters with each pooling block:

```
Block 1: 32 filters  (learn edges and gradients)
Block 2: 64 filters  (learn textures and shapes)
Block 3: 128 filters (learn object parts)
```

---

### Flattening and Dense Layers

After the convolutional blocks have reduced a 28×28 image to, say, a 7×7×64 tensor, we need to transition to the classification stage. The `Flatten` layer reshapes this 3D tensor into a 1D vector:

```
7 × 7 × 64 = 3,136 values → [3136-dimensional vector]
```

This vector then passes through one or more `Dense` (fully connected) layers. Unlike convolutional layers that share weights spatially, dense layers connect every input to every output neuron — they have the freedom to learn arbitrary combinations of the detected features, regardless of spatial position.

**Typical classification head:**
```python
Flatten()          # 3D → 1D
Dense(128, activation='relu')  # learn feature combinations
Dropout(0.5)       # regularization
Dense(10, activation='softmax')  # output: probabilities for 10 classes
```

The final `Dense` layer must have exactly as many neurons as there are classes, and use `softmax` to convert raw scores to probabilities.

---

### Keras Sequential API

The Sequential API is the simplest way to build CNNs when your architecture is a linear stack of layers (no branches, no shared layers). You add layers one by one:

```python
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
```

**When to use Sequential:** Simple architectures, learning, prototyping, architectures with a single input and single output.

**When NOT to use Sequential:** Multiple inputs, multiple outputs, residual/skip connections, shared layers, attention mechanisms.

---

### Keras Functional API

The Functional API gives you full control over the computation graph. Each layer is a callable that takes a tensor and returns a tensor:

```python
inputs = keras.Input(shape=(28, 28, 1))
x = keras.layers.Conv2D(32, (3,3), activation='relu')(inputs)
x = keras.layers.MaxPooling2D(2,2)(x)
x = keras.layers.Conv2D(64, (3,3), activation='relu')(x)
x = keras.layers.MaxPooling2D(2,2)(x)
x = keras.layers.Flatten()(x)
x = keras.layers.Dense(128, activation='relu')(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

This looks more verbose for simple models, but it enables architectures with branches (ResNet, Inception) and multiple inputs/outputs.

---

### Reading model.summary()

`model.summary()` is one of the most useful debugging tools when building CNNs. It shows:

```
Model: "sequential"
_________________________________________________________________
Layer (type)        Output Shape         Param #
=================================================================
conv2d (Conv2D)     (None, 26, 26, 32)   320
max_pooling2d       (None, 13, 13, 32)   0
conv2d_1 (Conv2D)   (None, 11, 11, 64)   18496
max_pooling2d_1     (None, 5, 5, 64)     0
flatten             (None, 1600)         0
dense (Dense)       (None, 128)          204928
dense_1 (Dense)     (None, 10)           1290
=================================================================
Total params: 225,034
```

**How to read Output Shape:**
- `(None, 26, 26, 32)` — batch size (None = flexible), height, width, channels
- After 3×3 conv (valid padding) on 28×28: output = 28-3+1 = **26**
- After MaxPool 2×2 on 26×26: output = 26/2 = **13**

**How to calculate parameter count:**
- Conv2D(32, 3×3): (3×3×1 + 1) × 32 = (9+1) × 32 = **320** params
  - 9 weights per filter × input channels × output filters, +1 bias per filter
- Dense(128): 1600 × 128 + 128 = **204,928** params

Notice how the Dense layers dominate the parameter count! This is why GAP (Global Average Pooling) is used in modern architectures to reduce FC parameters.

---

### LeNet-5: The Classic CNN Architecture

LeNet-5, designed by Yann LeCun in 1989 and published in 1998, was the first successful CNN applied to real-world problems (handwritten digit recognition for the US Postal Service). Its architecture established the template that all modern CNNs still follow:

```
INPUT: 32×32 grayscale image
  ↓
C1: Conv2D(6, 5×5, tanh)    → 28×28×6
  ↓
S2: AveragePooling2D(2×2)   → 14×14×6
  ↓
C3: Conv2D(16, 5×5, tanh)   → 10×10×16
  ↓
S4: AveragePooling2D(2×2)   →  5×5×16
  ↓
Flatten                      → 400
  ↓
F5: Dense(120, tanh)         → 120
  ↓
F6: Dense(84, tanh)          → 84
  ↓
OUTPUT: Dense(10, softmax)   → 10
```

The main differences from modern CNNs: it used tanh instead of ReLU (ReLU wasn't standard until AlexNet in 2012), average pooling instead of max pooling, and 5×5 filters instead of the now-standard 3×3 filters.

---

### Compilation and Training

Before training, you must `compile` the model with three components:

1. **Optimizer** — algorithm for updating weights (Adam, SGD, RMSprop)
2. **Loss function** — measures prediction error
   - `sparse_categorical_crossentropy` when labels are integers (0, 1, 2...)
   - `categorical_crossentropy` when labels are one-hot encoded
   - `binary_crossentropy` for binary classification
3. **Metrics** — what to track during training (accuracy is standard)

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

Then `model.fit()` runs the training loop, returning a `history` object containing loss and accuracy for each epoch.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Conv→ReLU→Pool Block | The fundamental CNN building block combining convolution, activation, and pooling | Repeated composition of these blocks builds increasingly abstract representations |
| Feature Extraction Stage | The convolutional portion of a CNN that learns spatial features | Provides the rich feature representations that enable classification |
| Classification Head | The fully-connected layers after flattening that predict class probabilities | Combines extracted features into final predictions |
| Flatten Layer | Converts multi-dimensional tensor to 1D vector | Bridges the convolutional and dense stages |
| Sequential API | Keras API for linear layer stacks | Simple, readable code for standard architectures |
| Functional API | Keras API for arbitrary computation graphs | Enables skip connections, multi-input/output models |
| model.summary() | Prints layer names, output shapes, and parameter counts | Essential debugging tool; shows shape flow through network |
| LeNet-5 | 1989 CNN architecture by Yann LeCun for digit recognition | Historical foundation; established the Conv-Pool-Dense template |
| Compile | Configuring model with optimizer, loss, and metrics before training | Required step that sets up the learning algorithm |
| sparse_categorical_crossentropy | Loss for integer-labeled multi-class problems | Most common loss function for classification |

---

## Code Reference

See [`code/lesson_08.py`](../code/lesson_08.py) for a fully runnable CNN on MNIST with model.summary(), training curves, predictions, and architectural comparison.

---

## Activities

1. **Shape Tracing Exercise:** For a CNN with the following layers on a 32×32×3 input, manually calculate the output shape after each layer: Conv2D(32, 3×3, valid), MaxPool(2×2), Conv2D(64, 3×3, valid), MaxPool(2×2), Flatten. Verify using `model.summary()`.

2. **Parameter Counting:** Given `Conv2D(64, (3,3))` applied to a tensor with 32 input channels, calculate the total number of trainable parameters (weights + biases). Then calculate the parameters for `Dense(256)` applied to a 512-dimensional input.

3. **Sequential to Functional Conversion:** Take the MNIST CNN from the code demo and rewrite it using the Functional API. Verify that `model.summary()` produces identical output for both versions.

4. **LeNet-5 Reimplementation:** Implement LeNet-5 using modern Keras. Replace tanh with ReLU, average pooling with max pooling, and adapt the input to MNIST's 28×28 format. Compare its accuracy to the modern CNN implementation.

5. **Depth Experiment:** Build three versions of a CNN for MNIST: one with 1 convolutional block, one with 2, and one with 3. Train all three for 10 epochs. Plot their learning curves on the same graph and compare final test accuracy. What do you observe about the benefit of additional depth?

---

## Review Questions

1. Explain the two main stages of a CNN (feature extraction and classification head). What does each stage learn, and why are both necessary?

2. What information does `model.summary()` provide? Walk through how you would calculate the parameter count for a `Conv2D(64, (3,3))` layer with 32 input channels.

3. What is the difference between the Sequential API and Functional API in Keras? When would you be forced to use the Functional API?

4. Why does filter depth typically increase with each convolutional block (e.g., 32→64→128)? What concept does this reflect about how features are organized hierarchically?

5. Describe the original LeNet-5 architecture. What two major changes did AlexNet (2012) make to this template, and how did those changes improve performance?

---

## Further Reading

- **"Gradient-Based Learning Applied to Document Recognition"** (LeCun et al., 1998) — The original LeNet-5 paper; foundational reading
- **"ImageNet Classification with Deep Convolutional Neural Networks"** (Krizhevsky et al., 2012) — The AlexNet paper that sparked the deep learning revolution
- **Keras Sequential and Functional API guides** — https://keras.io/guides/ — Official tutorial with side-by-side comparisons
- **"A Guide to Convolution Arithmetic for Deep Learning"** (Dumoulin & Visin, 2016) — Rigorous guide to shape calculations with visual animations
- **CS231n CNN Architecture notes** — https://cs231n.github.io/convolutional-networks/#architectures — Survey of landmark CNN architectures with analysis
