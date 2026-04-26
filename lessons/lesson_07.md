# Lesson 7: Pooling Layers

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the purpose of pooling layers in a CNN architecture
- Distinguish between max pooling, average pooling, and global average pooling
- Calculate the output dimensions of a feature map after pooling
- Implement pooling layers manually with NumPy and in Keras
- Describe how pooling contributes to translation invariance
- Understand when pooling may be replaced by strided convolutions

---

## Detailed Explanation

### What Is Pooling?

After a convolutional layer creates feature maps — each one highlighting a specific pattern in the input image — the network typically needs to **reduce the spatial size** of those maps before the next processing stage. This reduction is called **downsampling**, and pooling layers are the classical mechanism for achieving it.

Pooling operates on each feature map independently using a small window (e.g., 2×2) that slides across the map. Unlike convolution, pooling has **no learnable parameters** — it simply aggregates the values within each window into a single output value. This makes pooling computationally cheap and deterministic.

Without any downsampling, feature maps from early convolutional layers remain very large. A 224×224 input with 64 filters would produce feature maps totalling 224 × 224 × 64 = 3.2 million values. After pooling by a factor of 2, that drops to 112 × 112 × 64 = 800K values — a 4× reduction. This matters for both memory and computation in later layers.

---

### Max Pooling

Max pooling takes the **maximum value** from each pooling window. It is by far the most commonly used form of pooling in CNNs.

**Step-by-step example with a 4×4 feature map and 2×2 pool, stride 2:**

```
Input Feature Map:        After 2×2 Max Pooling:
┌───┬───┬───┬───┐         ┌───┬───┐
│ 1 │ 3 │ 2 │ 4 │         │ 3 │ 4 │   ← max(1,3,5,6)=6? No: top-left window
│ 5 │ 6 │ 7 │ 8 │         │ 8 │ 9 │     top-left: max(1,3,5,6)=6
├───┼───┼───┼───┤    →    └───┴───┘     top-right: max(2,4,7,8)=8? =8
│ 2 │ 1 │ 9 │ 3 │                       bottom-left: max(2,1,0,4)= ...
│ 0 │ 4 │ 2 │ 1 │
└───┴───┴───┴───┘

Windows:
Top-left   [1,3,5,6]   → max = 6
Top-right  [2,4,7,8]   → max = 8
Bot-left   [2,1,0,4]   → max = 4
Bot-right  [9,3,2,1]   → max = 9

Result:
┌───┬───┐
│ 6 │ 8 │
│ 4 │ 9 │
└───┴───┘
```

**Why max pooling works:** It retains the most prominent feature activation in each region. If a filter detects an edge anywhere in a 2×2 region, max pooling preserves that detection regardless of its exact position within the window.

**Hyperparameters:**
- **Pool size:** The window dimensions, typically 2×2
- **Stride:** How far the window moves each step, typically equal to pool size (no overlap)
- **Padding:** Usually 'valid' (no padding) for pooling

---

### Average Pooling

Average pooling replaces the maximum with the **mean** of the values in each window. This preserves more background information and provides a smoother, less aggressive downsampling.

```
Input window:     [1, 3, 5, 6]
Average:          (1 + 3 + 5 + 6) / 4 = 3.75
```

Average pooling is less common than max pooling in the middle of a network but sees significant use in specific contexts (e.g., in inception modules and in global pooling).

---

### Global Average Pooling (GAP)

Global average pooling reduces an **entire feature map to a single scalar value** — the average of all its values. If you have a feature map of shape (7, 7, 512), global average pooling produces a vector of shape (512,) — one number per channel.

```
Feature Map (7×7):         Global Average Pool:
┌──────────────┐           ┌───┐
│  all 49      │   →       │avg│  (single value)
│  values      │           └───┘
└──────────────┘
```

**Why GAP is powerful:**
- It eliminates the need for large Dense layers after the convolutional backbone
- It dramatically reduces overfitting (fewer parameters)
- It makes the network accept **any input size**, since the spatial dimensions are collapsed
- It has become the standard in modern architectures (ResNet, MobileNet, EfficientNet all use GAP)

---

### Why Pooling Helps

1. **Reduces spatial dimensions:** Cuts height and width by the stride factor, reducing computation in later layers
2. **Reduces parameter count:** Smaller feature maps mean smaller Dense layers
3. **Translation invariance:** A feature detected slightly off-center in the original image is still detected after pooling — the exact position becomes less important
4. **Implicit regularization:** By discarding fine spatial detail, pooling forces the network to focus on higher-level patterns
5. **Controls receptive field growth:** Each pooling layer doubles the effective receptive field of subsequent convolutions

---

### Translation Invariance via Pooling

Translation invariance means the network's output changes little if the input object moves slightly. Consider a vertical edge detector: if the edge is at pixel (10, 10) or at pixel (11, 10), both activations fall in the same 2×2 pooling window, and max pooling produces the same output. This local invariance accumulates across multiple pooling layers, making the network increasingly position-agnostic.

**Important nuance:** CNNs are not fully translation invariant — they are only locally invariant within each pooling window. Large shifts in the input can change the output. True full translation equivariance requires different architectures.

---

### Output Size Calculation

Given an input feature map of size H × W and a pooling layer with pool size P and stride S:

```
Output Height = floor((H - P) / S) + 1
Output Width  = floor((W - P) / S) + 1
```

**Example:** 28×28 input, 2×2 pool size, stride 2:
```
Output = floor((28 - 2) / 2) + 1 = floor(13) + 1 = 14
→ Output: 14×14
```

After three such pooling layers on a 224×224 input:
```
224 → 112 → 56 → 28
```

---

### Common Mistakes

- **Thinking pooling always helps:** In some tasks (segmentation, detection), aggressive pooling destroys spatial precision needed for the output. Architectures like U-Net preserve spatial resolution and only pool when necessary.
- **Choosing the wrong stride:** Setting stride = 1 with pooling size = 2 creates overlapping windows — more information retained but slower downsampling.
- **Pooling on tiny feature maps:** Applying 2×2 pooling to a 2×2 map reduces it to 1×1 — often fine, but can destroy too much information if done too early.
- **Forgetting GAP replaces Flatten:** Many students add both GAP and Flatten — use only one at the end of the convolutional backbone.

---

### Spatial Pyramid Pooling (SPP)

A brief but important concept: Spatial Pyramid Pooling pools a feature map at multiple scales simultaneously (e.g., 4×4, 2×2, and 1×1) and concatenates the results. This allows the network to accept inputs of **varying spatial sizes** while producing a fixed-length output vector. SPP was introduced in the SPPNet paper (He et al., 2014) and is used in object detection frameworks.

---

### Modern Trend: Strided Convolutions Replacing Pooling

Many modern architectures (e.g., All-Convolutional Networks, some ResNet variants) have replaced pooling layers with **strided convolutions** — convolutions with stride 2 instead of stride 1. This downsamples the feature map just like pooling, but the learned filters can determine *how* to downsample, rather than using a fixed max or average rule. The trade-off is more parameters, but the network has more flexibility.

The 2015 paper "Striving for Simplicity" by Springenberg et al. demonstrated that competitive performance could be achieved using only strided convolutions, sparking this trend.

---

### Effect on Feature Map Dimensions

| Layer             | Output Shape (example) |
|-------------------|------------------------|
| Input             | 28 × 28 × 1            |
| Conv2D (32, 3×3)  | 26 × 26 × 32           |
| MaxPool (2×2, s2) | 13 × 13 × 32           |
| Conv2D (64, 3×3)  | 11 × 11 × 64           |
| MaxPool (2×2, s2) | 5 × 5 × 64             |
| GlobalAvgPool     | 64                     |
| Dense (10)        | 10                     |

---

## Key Concepts Table

| Term                   | Definition                                                                  |
|------------------------|-----------------------------------------------------------------------------|
| Downsampling           | Reducing the spatial size of feature maps                                   |
| Max pooling            | Selecting the maximum value in each pooling window                          |
| Average pooling        | Computing the mean of values in each pooling window                         |
| Global average pooling | Reducing each feature map to a single scalar (its spatial average)          |
| Translation invariance | Insensitivity to small positional shifts of the input                       |
| Receptive field        | The region of the input image that influences a single neuron's output      |
| Stride                 | The number of pixels the pooling window moves between each computation      |
| Spatial Pyramid Pooling| Multi-scale pooling to handle variable input sizes                          |

---

## Code Reference

```python
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D

# Max pooling with 2x2 window, stride 2
pool = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

# Average pooling
avg_pool = AveragePooling2D(pool_size=(2, 2), strides=(2, 2))

# Global average pooling (replaces Flatten before Dense layers)
gap = GlobalAveragePooling2D()

# Manual max pooling with NumPy
import numpy as np

def max_pool_2d(feature_map, pool_size=2, stride=2):
    H, W = feature_map.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = feature_map[i*stride:i*stride+pool_size,
                                 j*stride:j*stride+pool_size]
            output[i, j] = np.max(window)
    return output
```

---

## Activities

1. **MaxPool vs AveragePool:** Using a `tf.keras.layers.MaxPooling2D` and a `tf.keras.layers.AveragePooling2D` layer (both pool_size=2, stride=2), apply each to the same randomly generated 8×8 feature map tensor. Print both output arrays and highlight cells where they differ.

2. **Dimension Tracing:** Build a small Keras CNN (2 Conv2D + MaxPooling2D blocks) for 32×32×3 inputs. Before running `model.summary()`, manually compute the spatial dimensions after each pooling layer. Verify your predictions against the summary output.
## Review Questions

1. What is the primary purpose of a pooling layer in a CNN?
2. How does max pooling differ from average pooling? When might you prefer one over the other?
3. Calculate the output size of a 32×32 feature map after applying 2×2 max pooling with stride 2.
4. What is global average pooling and why is it preferred over Flatten in modern architectures?
5. Explain how max pooling contributes to translation invariance.
6. What is the "dying pooling" mistake when applying pooling to very small feature maps?
7. Why are strided convolutions considered an alternative to pooling layers?
8. In spatial pyramid pooling, what problem does it solve that standard pooling cannot?

---

## Further Reading

- [CS231n: Convolutional Neural Networks — Pooling Layer](https://cs231n.github.io/convolutional-networks/#pool)
- [Network in Network — Min Lin et al. (introduces GAP)](https://arxiv.org/abs/1312.4400)
- [Striving for Simplicity — Springenberg et al., 2015](https://arxiv.org/abs/1412.6806)
- [SPPNet — He et al., 2014](https://arxiv.org/abs/1406.4729)
- [Deep Learning Book, Chapter 9 — Convolutional Networks](https://www.deeplearningbook.org/contents/convnets.html)
