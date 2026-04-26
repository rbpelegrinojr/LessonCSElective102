# Lesson 8: Building Your First Complete CNN

## Learning Objectives

By the end of this lesson, you will be able to:

- Describe the anatomy of a complete CNN from input to output
- Build a standard convolutional block: Conv2D → BatchNorm → ReLU → MaxPool
- Implement LeNet-5, the first modern CNN, in Keras
- Read and interpret a Keras model summary
- Choose appropriate filter counts, kernel sizes, and input shapes
- Compile a CNN and perform a complete forward pass

---

## Detailed Explanation

### The Anatomy of a Complete CNN

A complete Convolutional Neural Network consists of two major components working together: a **feature extraction backbone** and a **classification head**.

The **backbone** is a sequence of convolutional blocks that progressively transform the raw pixel input into rich, abstract feature representations. Early blocks learn low-level features like edges and corners. Middle blocks learn textures and parts. Late blocks learn high-level semantic concepts like "wheel" or "face."

The **classification head** takes the final feature representations from the backbone and maps them to class probabilities. It consists of a flattening or global averaging step followed by one or more dense (fully connected) layers, ending with a softmax output.

```
Input Image
     │
     ▼
┌─────────────────────┐
│   Conv Block 1      │  (edges, gradients)
│   Conv→BN→ReLU→Pool │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv Block 2      │  (textures, simple shapes)
│   Conv→BN→ReLU→Pool │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Conv Block 3      │  (object parts, high-level)
│   Conv→BN→ReLU→Pool │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│ Classification Head │
│ Flatten/GAP → Dense │
│   → Softmax         │
└─────────────────────┘
         │
         ▼
   Class Probabilities
```

---

### The Standard Convolutional Block

The building block of modern CNNs follows a consistent pattern:

**Conv2D → Batch Normalization → ReLU → MaxPooling2D**

Each component has a specific role:

1. **Conv2D:** Learns spatial features using learnable filters. Produces a linear combination of neighboring pixels.
2. **Batch Normalization:** Normalizes the output of the convolution across the batch, stabilizing training, allowing higher learning rates, and acting as a regularizer. It was introduced by Ioffe and Szegedy in 2015 and dramatically accelerated training of deep networks.
3. **ReLU:** Introduces non-linearity. Applied after normalization so it acts on normalized values centered near zero, where ReLU is most effective.
4. **MaxPooling2D:** Downsamples the feature map by 2×, reducing computation and building spatial invariance.

In practice, you may see variations — some architectures skip BatchNorm, some use it before ReLU, some use strided convolutions instead of pooling. But the Conv → BN → ReLU → Pool pattern is an excellent starting point.

---

### LeNet-5: The First Modern CNN (1998)

LeNet-5 was designed by Yann LeCun and published in 1998 for recognizing handwritten digits. Despite being over 25 years old, it remains a perfect teaching example because it is small enough to understand completely yet powerful enough to achieve excellent results on MNIST.

**LeNet-5 Architecture:**

```
Input: 32×32×1 (grayscale image)
  │
  ▼
C1: Conv2D(6 filters, 5×5, no padding) → 28×28×6
  │
  ▼
S2: AveragePooling2D(2×2, stride 2)   → 14×14×6
  │
  ▼
C3: Conv2D(16 filters, 5×5, no padding) → 10×10×16
  │
  ▼
S4: AveragePooling2D(2×2, stride 2)   → 5×5×16
  │
  ▼
Flatten                               → 400
  │
  ▼
F5: Dense(120, activation='tanh')
  │
  ▼
F6: Dense(84, activation='tanh')
  │
  ▼
Output: Dense(10, activation='softmax') → 10
```

Note that the original LeNet-5 used average pooling and tanh activations. Modern implementations typically replace these with max pooling and ReLU.

**Parameter count:**
- C1: (5×5×1 + 1) × 6 = 156 parameters
- C3: (5×5×6 + 1) × 16 = 2,416 parameters
- F5: (400 + 1) × 120 = 48,120 parameters
- F6: (120 + 1) × 84 = 10,164 parameters
- Output: (84 + 1) × 10 = 850 parameters
- **Total: ~61,706 parameters** — tiny by modern standards!

---

### Choosing the Number of Filters

A common and effective strategy is to **start with a small number of filters and double with each block:**

```
Block 1: 32 filters
Block 2: 64 filters
Block 3: 128 filters
Block 4: 256 filters (if needed)
```

This works because:
- Early layers detect simple, low-level features (fewer needed)
- Deeper layers detect complex, combinatorial features (more needed)
- Doubling keeps the total computation roughly constant because spatial dimensions halve at each pooling step

For small datasets (like MNIST, CIFAR-10), start with 32 filters. For large, complex datasets (ImageNet), start with 64.

---

### Choosing Kernel Size

The 3×3 kernel is the dominant choice in modern CNNs (VGGNet popularized this in 2014). Here is why:

- **3×3** captures local spatial relationships efficiently. Two stacked 3×3 convolutions have the same receptive field as one 5×5, but fewer parameters and more non-linearity.
- **5×5 and 7×7** are used in the first layer only (e.g., VGG, ResNet) to capture larger initial context from raw pixels.
- **1×1** convolutions reduce channel dimensions without touching spatial dimensions — used for bottleneck layers.

**Rule of thumb:** Use 3×3 kernels everywhere except the very first layer, where 5×5 or 7×7 may provide better initial features.

---

### Input Shape Considerations

In Keras, the input shape is specified as (height, width, channels):
- Grayscale images: (28, 28, 1) or (32, 32, 1)
- RGB images: (32, 32, 3) or (224, 224, 3)
- Always normalize pixel values to [0, 1] by dividing by 255.0

The network's depth and filter counts should be proportional to input size. Very small inputs (28×28) cannot support more than 2–3 pooling operations before the feature maps become 1×1.

---

### Reading a Model Summary

Keras's `model.summary()` prints three columns for each layer:
- **Layer (type):** The layer class and name
- **Output Shape:** The shape of data leaving that layer (None = batch dimension)
- **Param #:** Number of trainable parameters in that layer

For a Conv2D(32, 3, 3) on a (28, 28, 1) input:
```
Output shape: (None, 26, 26, 32)
Params: (3×3×1 + 1) × 32 = 320
```
The `+1` accounts for the bias term per filter.

---

### Compilation: Choosing Optimizer and Loss

For classification tasks:
- **Binary classification (2 classes):** loss='binary_crossentropy', final activation='sigmoid'
- **Multi-class (N classes, integer labels):** loss='sparse_categorical_crossentropy', final activation='softmax'
- **Multi-class (N classes, one-hot labels):** loss='categorical_crossentropy', final activation='softmax'

**Optimizer:**
- `adam` is the universal default (adaptive learning rates, momentum)
- `sgd` with momentum for fine-tuned training after initial convergence

**Metrics:**
- `accuracy` is always appropriate for balanced classification datasets

```python
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
```

---

### Sequential API vs Functional API

**Sequential API:** Layers stacked linearly. Simple, clean, but inflexible.
```python
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dense(10, activation='softmax')
])
```

**Functional API:** Layers connected explicitly as a graph. Supports branching, skip connections, multiple inputs/outputs.
```python
inputs = tf.keras.Input(shape=(28, 28, 1))
x = layers.Conv2D(32, 3, activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
outputs = layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)
```

Use Sequential for simple architectures. Switch to Functional when you need ResNet-style skip connections or multi-branch designs.

---

### Total Trainable Parameters Calculation

For any layer, trainable parameters = (kernel_size × kernel_size × in_channels + 1_bias) × out_channels

```
Conv2D(32, 3, input_channels=1):  (3×3×1 + 1) × 32  = 320
Conv2D(64, 3, input_channels=32): (3×3×32 + 1) × 64 = 18,496
Dense(128, input=576):            (576 + 1) × 128    = 73,856
Dense(10, input=128):             (128 + 1) × 10     = 1,290
Total: 93,962 parameters
```

BatchNorm layers add 4 parameters per channel (gamma, beta, moving mean, moving variance — though the last two are non-trainable).

---

## Key Concepts Table

| Term                   | Definition                                                                  |
|------------------------|-----------------------------------------------------------------------------|
| Convolutional block    | Standard unit: Conv2D → BatchNorm → ReLU → MaxPool                         |
| Classification head    | Flatten/GAP + Dense layers that map features to class probabilities         |
| LeNet-5                | First successful modern CNN, designed by LeCun et al. in 1998              |
| Batch Normalization    | Normalizes layer outputs across the batch to stabilize training             |
| Model summary          | Keras output showing layer types, output shapes, and parameter counts       |
| Sequential API         | Keras interface for building linear layer stacks                            |
| Functional API         | Keras interface for building arbitrary computation graphs                   |
| Kernel                 | The learnable weight matrix in a Conv2D layer                               |

---

## Code Reference

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# Build a 3-block CNN with Sequential API
model = models.Sequential([
    # Block 1
    layers.Conv2D(32, (3, 3), padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    # Block 2
    layers.Conv2D(64, (3, 3), padding='same'),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D((2, 2)),

    # Classification Head
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()
```

---

## Activities

1. **LeNet-5 in Keras:** Implement LeNet-5 using `keras.Sequential` (Conv2D(6,5,tanh) → AvgPool → Conv2D(16,5,tanh) → AvgPool → Flatten → Dense(120,tanh) → Dense(84,tanh) → Dense(10,softmax)). Run `model.summary()` and verify the total parameter count is approximately 61 706.

2. **Architecture Modification:** Start with the tiny 3-block CNN from `code/lesson_08.py`. Double the number of filters in every Conv2D layer. Compare total parameter count and validation accuracy after 3 epochs on MNIST to the original architecture.
## Review Questions

1. What are the two main components of a complete CNN, and what role does each play?
2. Why is Batch Normalization placed between Conv2D and the activation function?
3. What makes LeNet-5 historically significant in the development of deep learning?
4. If you have a 28×28×1 input and apply Conv2D(32, 5×5, valid padding) followed by MaxPooling2D(2×2), what is the output shape?
5. Why is the 3×3 kernel preferred over 5×5 in modern CNNs?
6. When should you use the Functional API instead of the Sequential API?
7. Calculate the trainable parameters in a Dense(256) layer that receives a flattened input of 7×7×64.
8. What compilation settings would you use for a 10-class classification problem with one-hot labels?

---

## Further Reading

- [LeCun et al. 1998 — Gradient-Based Learning Applied to Document Recognition](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)
- [VGGNet — Simonyan & Zisserman, 2014](https://arxiv.org/abs/1409.1556)
- [Batch Normalization — Ioffe & Szegedy, 2015](https://arxiv.org/abs/1502.03167)
- [Keras Functional API Guide](https://keras.io/guides/functional_api/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/convolutional-networks/)
