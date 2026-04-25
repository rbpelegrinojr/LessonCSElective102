# Lesson 05: The Convolution Operation Deep Dive

## Learning Objectives
- Define convolution mathematically and describe it geometrically as a sliding window dot product
- Manually calculate the output of a 2D convolution for a given input matrix and kernel
- Distinguish between "valid" and "same" padding and calculate output dimensions for each
- Define stride and explain its effect on the spatial dimensions of the feature map
- Explain how multiple filters produce multiple feature maps and why this is necessary
- Identify the patterns detected by specific well-known kernels (Sobel, Gaussian, sharpening)

---

## Detailed Explanation

### What Is Convolution?

In signal processing, convolution is an operation that expresses how the shape of one function is modified by another. In image processing — and in CNNs — we use a discrete 2D version of convolution: we slide a small matrix (the **kernel** or **filter**) over a larger matrix (the image) and at each position compute the sum of element-wise products.

The result is a new matrix called the **feature map** (or **activation map**), where each value tells you "how strongly does this kernel's pattern appear at this location in the image?"

Let us define the operation precisely:

```
Given:
  Input image:     I  (of size H × W)
  Filter (kernel): K  (of size k × k, typically 3×3 or 5×5)

Output feature map value at position (i, j):
  FM[i, j] = ΣᵣΣ꜀  I[i+r, j+c] × K[r, c]

  Where r and c index over the filter dimensions.
```

In plain English: place the filter on top of the image at position (i, j). Multiply each filter element by the overlapping image pixel. Sum all those products. That sum is the feature map value at (i, j). Then slide the filter one step to the right and repeat.

### A Manual Convolution Example

Let us work through a complete 3×3 convolution by hand:

```
Input image (5×5):          Filter / Kernel (3×3):
 1  2  3  4  5               0  1  0
 6  7  8  9 10               1  1  1
11 12 13 14 15               0  1  0
16 17 18 19 20
21 22 23 24 25

Computing output at position (0, 0) — top-left corner:
  (uses the 3×3 region starting at row 0, col 0)

  Overlap:     Image patch      Filter
                1  2  3          0  1  0
                6  7  8    ×     1  1  1
               11 12 13          0  1  0

  Products:
   1×0=0    2×1=2    3×0=0
   6×1=6    7×1=7    8×1=8
  11×0=0   12×1=12  13×0=0

  Sum = 0+2+0 + 6+7+8 + 0+12+0 = 35

Output at position (0,0) = 35

Computing output at position (0, 1) — slide right by 1:
  Image patch:   2  3  4
                 7  8  9
                12 13 14

  Products with same filter:
   2×0=0    3×1=3    4×0=0
   7×1=7    8×1=8    9×1=9
  12×0=0   13×1=13  14×0=0

  Sum = 0+3+0 + 7+8+9 + 0+13+0 = 40

Output at position (0,1) = 40
```

The filter above (0,1,0 / 1,1,1 / 0,1,0) is a "plus-shaped" filter that sums the centre pixel and its four direct neighbours. It is a simple smoothing filter. Different filter weights detect different patterns.

### Output Dimensions: The Dimension Formula

When we slide a k×k filter over an H×W image, how large is the output feature map?

**With VALID padding (no padding):**
The filter cannot extend beyond the image boundary, so it only fits in positions where it fully overlaps the image.

```
Output height = H - k + 1
Output width  = W - k + 1

Example: 5×5 image, 3×3 filter, valid padding:
  Output height = 5 - 3 + 1 = 3
  Output width  = 5 - 3 + 1 = 3
  Output shape: 3×3
```

Each application of a convolutional layer reduces spatial dimensions. After many layers, the feature map shrinks to nothing. This is sometimes desired (we want the network to become spatially compact), but we need to control the rate of shrinkage.

**With SAME padding (zero-padding):**
We add zeros around the border of the image so that the output has the same spatial dimensions as the input.

```
For a k×k filter to produce same-size output:
  Padding p = (k - 1) / 2    (on each side, for odd-sized filters)

  3×3 filter: p = (3-1)/2 = 1 zero on each side
  5×5 filter: p = (5-1)/2 = 2 zeros on each side

Example: 5×5 image, 3×3 filter, same padding:
  Padded input size: (5+2)×(5+2) = 7×7
  Output height = 7 - 3 + 1 = 5
  Output width  = 7 - 3 + 1 = 5
  Output shape: 5×5  (same as input)
```

"Same" padding is named so because the output spatial size is the same as the input. This is the default in most CNN architectures when we want to control where spatial downsampling occurs (in pooling layers rather than convolutions).

### Stride: Controlling the Sliding Step

By default, the filter slides one pixel at a time (stride = 1). Increasing the stride causes the filter to jump multiple pixels between positions, producing a smaller output feature map.

```
General dimension formula with stride s and padding p:
  Output size = floor((H + 2p - k) / s) + 1

Examples (H=28, k=3):
  Stride 1, no padding (p=0):  floor((28+0-3)/1) + 1 = 26
  Stride 2, no padding (p=0):  floor((28+0-3)/2) + 1 = 13
  Stride 1, same padding:      floor((28+2-3)/1) + 1 = 28
  Stride 2, same padding:      floor((28+2-3)/2) + 1 = 14
```

Stride 2 roughly halves the spatial dimensions, similar to a 2×2 pooling layer. Some modern architectures (like ResNet's initial layer) use strided convolutions instead of pooling to downsample.

### Multiple Filters: The Depth Dimension

A single filter can only detect one pattern. To detect many patterns simultaneously (edges in all directions, colour blobs, corners, etc.), we apply many filters in parallel. If we apply 32 filters to a single input, we produce 32 feature maps — one per filter.

```
Input:   (H, W, C_in)     e.g., (28, 28, 1)  for grayscale
Filters: (k, k, C_in, C_out)  e.g., (3, 3, 1, 32)
Output:  (H, W, C_out)    e.g., (28, 28, 32) with same padding

The output "depth" (C_out) equals the number of filters.
```

For colour images (3 channels), each filter has depth 3 — it looks at a 3×3 region across all three colour channels simultaneously:

```
Colour input (H, W, 3):

  Red channel:     ┌───┐
  Green channel:   │   │  ← filter volume: 3×3×3 = 27 values
  Blue channel:    └───┘

  Each filter produces ONE feature map by computing the weighted sum
  across the spatial AND channel dimensions simultaneously.
```

The parameter count for a convolutional layer:
```
  parameters = (k × k × C_in + 1) × C_out
                └─────────────┘   └──────┘
                 per-filter params  number of filters
                 (including bias)

  Example: 32 filters of size 3×3 applied to RGB (C_in=3):
  parameters = (3 × 3 × 3 + 1) × 32 = (27 + 1) × 32 = 896
```

### Kernels and What They Detect

Different filter weights detect different patterns. Here are some classical image processing kernels:

**Horizontal Edge Detection (Sobel filter):**
```
  -1  -2  -1         Detects horizontal edges (bright above dark).
   0   0   0         Activates strongly where pixel values
   1   2   1         change vertically (gradient in y-direction).
```

**Vertical Edge Detection (Sobel filter):**
```
  -1  0  1           Detects vertical edges (bright left of dark).
  -2  0  2           Activates at vertical transitions
  -1  0  1           (gradient in x-direction).
```

**Identity (no-op):**
```
  0  0  0            Output equals input.
  0  1  0            Used in skip/residual connections.
  0  0  0
```

**Gaussian Blur:**
```
  1  2  1
  2  4  2   × (1/16)   Smooths image, removes noise.
  1  2  1               Each output pixel is a weighted average
                        of its neighbourhood.
```

**Sharpening:**
```
   0  -1   0            Emphasises edges, increases apparent sharpness.
  -1   5  -1            Subtracts a blurred version from the original.
   0  -1   0
```

**The key insight for CNNs:** The network *learns* the kernel values through backpropagation. You do not hand-design kernels — you specify how many filters you want and what size, and training finds the values that minimise the loss. In practice, the learned filters often resemble Gabor filters (oriented edge detectors) in the first layer and increasingly abstract pattern detectors in deeper layers.

### Visualising Feature Maps: What Does the Network See?

After training, each convolutional filter has learned to respond to a specific pattern. To understand what a filter detects, we can:

1. **Visualise the filter weights directly** — for 3×3 greyscale filters, just display the 3×3 grid as an image.
2. **Display feature maps** — pass an image through the network and display the output of each filter at a given layer. Bright regions show where the filter's pattern was detected.

Zeiler and Fergus (2013) famously showed that the first layer of deep CNNs learns Gabor-like filters (oriented edges), the second layer learns corners and curves, and deeper layers learn faces, text, and complex textures. This directly validates the hierarchical local-to-global feature learning hypothesis.

### Convolution vs Cross-Correlation

A strict mathematical note: what CNNs actually perform is **cross-correlation**, not true convolution. In true convolution, the kernel is flipped 180° before the sliding dot product. In cross-correlation, it is not. Because neural network filters are learned (not hand-designed), the flip makes no difference in practice — the network can simply learn the flipped version. By convention, we continue to call the operation "convolution" in deep learning.

### Efficiency: Why Convolution Is Computationally Tractable

The key efficiency advantage of convolutions over dense layers:

```
Dense layer:  O(H × W × C_in × N_neurons)  parameters AND multiply-adds
Conv layer:   O(k × k × C_in × C_out)      parameters
              O(H × W × k × k × C_in × C_out)  multiply-adds
              (same multiply-adds as dense, but FAR fewer parameters)
```

CNNs do NOT necessarily compute fewer operations than dense networks — they compute different operations more intelligently by reusing the same filter weights across spatial positions. The savings are in parameters (and therefore in memory and in how much training data is needed), not necessarily in raw computation.

---

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| **Convolution** | Sliding a filter over an input and computing dot products at each position to produce a feature map | The core operation of CNNs; how spatial patterns are detected efficiently |
| **Kernel / Filter** | A small matrix of learnable weights applied at every position in the input | Contains the learned "detector" for one specific pattern |
| **Feature Map** | The output of applying one filter to the entire input; encodes where the pattern was detected | The data structure passed between convolutional layers |
| **Stride** | The number of pixels the filter moves between positions | Controls the spatial downsampling rate of the convolution |
| **Valid Padding** | No padding; output is smaller than input by (k-1) pixels per side | Reduces spatial dimensions; used when shrinkage is acceptable |
| **Same Padding** | Zero-padding added so output has same spatial size as input | Preserves spatial dimensions through convolution layers |
| **Depth / Channels** | The number of filters (and hence feature maps) in a layer | Determines the richness of the feature representation |
| **Receptive Field** | The region of the input image a single output neuron "sees" | Grows with depth; deeper neurons detect larger patterns |
| **Sobel Filter** | A hand-designed kernel for detecting edges in horizontal or vertical directions | Illustrates that kernels encode specific detectors; CNNs learn analogous filters |
| **Cross-Correlation** | The actual mathematical operation performed by CNNs (convolution without kernel flip) | Understanding the exact operation prevents confusion with signal processing convolution |

---

## Code Reference

See `code/lesson_05.py` for runnable demos: manual numpy convolution, Keras Conv2D usage, feature map visualisation for multiple filters, and application of Sobel and Gaussian kernels.

---

## Activities

1. **Manual Convolution:** Given the 5×5 input matrix and 3×3 kernel shown in this lesson, manually calculate ALL output values (you have two already). Fill out the complete 3×3 output matrix. Verify your result by implementing the same convolution in numpy using array slicing.

2. **Padding Experiment:** In `lesson_05.py`, apply a Conv2D layer to an MNIST image with `padding='valid'`, then with `padding='same'`. Print the output shape in both cases. Confirm that same-padding preserves spatial dimensions and valid-padding reduces them.

3. **Stride Comparison:** Apply a 3×3 Conv2D with stride=1, stride=2, and stride=3 to a 28×28 MNIST image. Calculate the expected output dimensions using the formula from this lesson, then verify by printing the actual output shapes. Plot the feature maps for all three strides side by side.

4. **Custom Kernel Application:** Using numpy and scipy (or tensorflow), manually apply the Sobel horizontal and vertical edge detection kernels to a CIFAR-10 image. Display the original image, the horizontal edge response, the vertical edge response, and the combined edge magnitude as a 2×2 grid.

5. **Filter Visualisation:** Train the CNN from `lesson_05.py` on MNIST. After training, extract the weights of the first convolutional layer (shape will be 3×3×1×32 for 32 filters). Display all 32 filters as a grid of 3×3 grayscale images. Do any of the learned filters resemble the edge detection kernels discussed in this lesson?

---

## Review Questions

1. Describe the convolution operation step by step. Given a 6×6 input and a 3×3 filter with valid padding and stride 1, what is the output shape? Show your calculation using the dimension formula.

2. What is the difference between "valid" and "same" padding? When would you choose each? Describe a scenario in a CNN architecture where each would be the appropriate choice.

3. A convolutional layer has 64 filters of size 5×5 applied to an input with 32 channels. How many learnable parameters does this layer have (include biases)? Show your calculation.

4. Explain why increasing the stride has a similar effect to pooling. What is gained and what is potentially lost by using strided convolutions instead of convolution + pooling?

5. If a single filter can only detect one type of pattern, how does a deep CNN end up being able to classify thousands of different object categories? Trace the answer through the concept of feature hierarchies and the composition of simple patterns into complex ones.

---

## Further Reading

- **"A guide to convolution arithmetic for deep learning" (Dumoulin & Visin, arXiv 2016)** — The definitive visual reference for convolution dimensions, padding, and stride with animations; freely available on arXiv.
- **"Understanding Convolutions" — Chris Olah's blog (colah.github.io)** — An intuitive mathematical explanation connecting signal processing convolution to the CNN operation with beautiful visualisations.
- **Feature Visualisation tool: TensorFlow Playground (playground.tensorflow.org)** — Interactive browser tool for visualising how neural network layers transform data; great for building intuition.
- **"Deep Visualization Toolbox" (Yosinski et al., 2015)** — Research demonstrating real-time visualisation of CNN feature maps, with video available on YouTube; directly shows what learned filters detect.
- **Keras documentation: Conv2D** — Full API documentation including all parameters (filters, kernel_size, strides, padding, activation, data_format) with practical examples.
