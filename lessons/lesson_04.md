# Lesson 04: From Dense Networks to CNNs

## Learning Objectives
- Quantify the parameter explosion that occurs when applying dense networks to high-resolution images
- Explain spatial invariance and why it is a desirable property for image classification
- Describe the concept of local pattern detection and explain why it motivates convolutional filters
- Define parameter sharing and explain how it reduces the number of learnable parameters in a CNN
- Compare the architecture of a dense network and a CNN side by side, including parameter counts
- Interpret the relationship between receptive fields and the hierarchy of learned features

---

## Detailed Explanation

### The Problem: Dense Networks and Images

In Lesson 3 we built a fully connected (dense) neural network and trained it on MNIST. It worked. But MNIST is an unusually forgiving dataset — tiny 28×28 grayscale images with clean white digits on black backgrounds. What happens when we try to apply dense networks to real-world colour images?

Let us calculate the parameter count for a single dense hidden layer on different image sizes:

```
Image: 28×28 grayscale (MNIST)
  Input neurons:  28 × 28 × 1  =    784
  Hidden layer:                    512 neurons
  Parameters:     784 × 512 + 512 = 401,920

Image: 224×224 RGB (ImageNet standard)
  Input neurons:  224 × 224 × 3  = 150,528
  Hidden layer:                    512 neurons
  Parameters:     150,528 × 512 + 512 = 77,071,360  (~77 million!)

Image: 1024×1024 RGB (typical smartphone photo)
  Input neurons:  1024 × 1024 × 3 = 3,145,728
  Hidden layer:                     512 neurons
  Parameters:     3,145,728 × 512 + 512 ≈ 1.6 BILLION (just layer 1!)
```

A network with 77 million parameters in the first layer alone faces several serious problems:

**1. Memory:** 77 million float32 values require ~308 MB of RAM just for one layer's weights — and we typically have many layers.

**2. Computation:** A forward pass requires 77 million multiplications for every single image, every single time. Training requires computing gradients for all those parameters too.

**3. Overfitting:** With so many parameters and a limited number of training images, the network has enormous capacity to memorise training examples rather than learning generalisable patterns. This is the curse of dimensionality made concrete.

**4. Structural blindness:** A dense layer treats every pixel as completely independent of every other pixel. It has no idea that pixel at position (50, 50) is spatially adjacent to pixel at (50, 51). This throws away a fundamental truth about images: nearby pixels are correlated, and local patterns matter.

### What Dense Layers Ignore: Spatial Structure

Images have spatial structure. A cat's eye looks like an eye regardless of whether it appears in the top-left corner or the bottom-right corner of an image. A circle with a dark centre looks like an eye in either location. A dense network, which applies different weights to every input position, cannot leverage this fact at all.

Consider this thought experiment: you train a dense network to recognise cats. The network learns that certain pixels in the centre-left region of a 224×224 image correspond to a cat's face. Now you show it a photo of the same cat where the camera is zoomed out slightly, so the cat's face is in the lower-right corner. From the dense network's perspective, the activations at the input nodes are completely different — the learned pattern no longer matches. The network may fail to recognise it as a cat.

This property — sensitivity to where an object appears in the image — is called **translation variance** (or spatial sensitivity). We want the opposite: **translation invariance** (or spatial invariance) — the ability to recognise the same pattern regardless of its position.

### Local Patterns: The Key Insight Behind CNNs

Human vision is not global. When you look at a face, your visual cortex first detects low-level local features: edges, gradients, corners. These combine into mid-level features: circles, curves, bars. These combine into high-level features: eyes, noses, mouths. Only then does your brain recognise "this is a face."

CNNs are explicitly designed around this hierarchical local-to-global structure. Instead of connecting every pixel to every neuron, a CNN applies small **filters** (also called **kernels**) that look at small local regions of the image — typically 3×3 or 5×5 pixels. Each filter learns to detect one specific local pattern.

```
Full image (7×7 pixels):        3×3 filter window:
┌─────────────────────┐         ┌─────────┐
│ 0  0  0  0  0  0  0 │         │ w1 w2 w3│
│ 0  0 255 255 255 0  │    →    │ w4 w5 w6│  slides across the image
│ 0 255   0   0 255 0 │         │ w7 w8 w9│
│ 0 255   0   0 255 0 │         └─────────┘
│ 0  0 255 255 255 0  │
│ 0  0  0  0  0  0  0 │
└─────────────────────┘
```

The filter slides across every possible position in the image, computing a dot product (weighted sum) at each position. The result is a new 2D grid of numbers called a **feature map**, which encodes where in the image the detected pattern was found.

### Parameter Sharing: The Game-Changer

Here is the critical insight: **the same filter weights are used at every position in the image**. A filter that learns to detect a vertical edge uses exactly the same 9 weights (for a 3×3 filter) whether it is looking at the top-left corner, the centre, or the bottom-right corner.

This is called **parameter sharing**, and it is the primary reason CNNs are so much more efficient than dense networks for images.

Compare parameter counts for detecting a set of 32 features:

```
Dense approach (per pixel, 224×224×3 input):
  Each of 32 "feature detectors" has its own weight for every input pixel
  Parameters: 150,528 × 32 = 4,816,896  (~4.8 million)

CNN approach (3×3 filter, 3 input channels):
  Each filter has only 3×3×3 = 27 weights + 1 bias, shared everywhere
  Parameters: 32 × (27 + 1) = 896   (!!!)
```

A single convolutional layer with 32 filters achieves feature detection across the entire image with just 896 parameters instead of nearly 5 million. This reduction is by a factor of ~5,000x.

### Spatial Invariance Through Parameter Sharing

Because the same filter weights are applied everywhere, a CNN naturally achieves a degree of translation invariance. If the network learns a filter that detects a cat's eye in the upper-left of one training image, that same filter will also activate when a cat's eye appears in the lower-right of a test image. The filter fires wherever its pattern is present, regardless of location.

This is not perfect translation invariance (pooling layers and data augmentation help further), but it is a vast improvement over dense networks and matches what we intuitively expect from an image recognition system.

### The CNN Architecture: An Overview

A typical CNN interleaves several types of layers:

```
Input Image
     ↓
[Conv Layer] — apply filters, produce feature maps, add spatial detail
     ↓
[Activation (ReLU)] — introduce non-linearity
     ↓
[Pooling Layer] — downsample, reduce spatial dimensions, increase invariance
     ↓
[Conv Layer] — detect higher-level combinations of lower-level features
     ↓
[Activation (ReLU)]
     ↓
[Pooling Layer]
     ↓
[Flatten] — convert 3D feature map tensor to 1D vector
     ↓
[Dense Layer] — combine all spatial features globally
     ↓
[Softmax Output] — class probabilities
```

Early convolutional layers detect simple features (edges, colour blobs). Deeper layers combine those simple features to detect complex patterns (shapes, object parts, textures). The final dense layers combine the spatially detected features to make the global classification decision.

### Receptive Fields and the Feature Hierarchy

The **receptive field** of a neuron is the region of the input image that influences its activation. A neuron in the first conv layer (with a 3×3 filter) has a 3×3 receptive field. A neuron two conv layers deep "sees" a larger region because its input (the first feature map) was itself computed from overlapping windows.

As you go deeper in a CNN, neurons have larger effective receptive fields and detect increasingly abstract patterns:

```
Layer 1:  3×3 receptive field   → detects edges, colour gradients
Layer 2:  ~7×7 receptive field  → detects corners, curves, texture patches  
Layer 3:  ~15×15 receptive field → detects eyes, wheels, leaves
Layer 4:  ~31×31 receptive field → detects faces, animals, vehicles
```

This hierarchy mirrors the structure of the mammalian visual cortex (V1, V2, V4, IT regions), which is why CNNs are considered biologically inspired even though they are not biological simulations.

### Dense vs CNN: A Direct Comparison

```
Property              Dense Network           CNN
─────────────────────────────────────────────────────────────
Connectivity          Every pixel to          Only local 3×3 region
                      every neuron            to each neuron
Parameter count       Very high               Much lower
Translation invariant No                      Approximately yes
Exploits spatial corr No                      Yes (by design)
Good for images?      Struggles above ~64×64  Designed for images
Good for tabular data Yes                     Overkill
Training speed        Slow for large images   Much faster
Overfitting risk      High (few samples)      Lower (fewer params)
```

This comparison makes clear why CNNs replaced dense networks as the standard architecture for image classification starting around 2012. CNNs are not just faster — they embody a fundamentally better inductive bias for image data.

---

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| **Parameter Explosion** | The exponential growth in parameter count when applying dense layers to large images | Motivates the need for CNNs over fully-connected networks for vision |
| **Spatial Invariance** | The ability of a model to recognise a pattern regardless of its position in the image | A fundamental requirement for robust image classifiers |
| **Local Pattern** | A visual feature defined over a small spatial region (e.g., an edge in a 3×3 patch) | CNNs explicitly exploit the locality of meaningful visual patterns |
| **Parameter Sharing** | Using the same filter weights at every spatial location in the image | The key mechanism that makes CNNs efficient; the same detector finds a pattern anywhere |
| **Feature Map** | The output of applying one filter across an entire image; encodes where a pattern was detected | The output of each convolutional layer; passed to the next layer as input |
| **Receptive Field** | The region of the input image that influences a particular neuron's activation | Grows with network depth; determines how large a pattern a neuron can detect |
| **Inductive Bias** | Assumptions built into an architecture that make it better suited for certain data types | CNNs have image-friendly inductive biases; dense networks do not |
| **Translation Invariance** | The property that shifting an object in the image does not change the classification | Achieved approximately by CNNs through parameter sharing and pooling |

---

## Code Reference

See `code/lesson_04.py` for runnable demos: side-by-side comparison of dense vs CNN architectures on MNIST, parameter counting, accuracy comparison, and translation invariance demonstration.

---

## Activities

1. **Parameter Calculator:** Use the formulas from this lesson to manually calculate the total parameter count for a network with: Conv(32, 3×3) → Conv(64, 3×3) → Dense(128) → Dense(10). Use input shape (28, 28, 1) and account for padding and pooling if applicable. Verify with `model.summary()`.

2. **Scaling Experiment:** Write code to build a dense network for three input sizes: 28×28, 64×64, and 128×128. Print the parameter count for each. Plot a bar chart showing how parameters grow with image size. At what size does the parameter count become infeasible for a standard laptop?

3. **Translation Test:** Take a CNN trained on MNIST. Create a modified test image by shifting a digit 5 pixels to the right (use numpy array operations). Compare the model's predictions on the original and shifted image. Does the CNN handle the shift better than a dense network? Repeat for a 10-pixel shift.

4. **Feature Map Intuition:** After training the CNN in `lesson_04.py`, add code to extract and display the feature maps from the first convolutional layer for a specific test image. How many feature maps are there (one per filter)? Describe what patterns you can visually observe in different feature maps.

5. **Architecture Hunt:** Research one real CNN architecture (VGG16, AlexNet, or ResNet-18) and document: the number of convolutional layers, the number of dense layers, the total parameter count, and the top-5 accuracy on ImageNet. Explain how the architectural choices reflect the principles discussed in this lesson.

---

## Review Questions

1. Explain why a dense network with a single hidden layer of 512 neurons becomes computationally impractical for a 224×224 colour image. Use exact parameter counts in your answer.

2. Define parameter sharing in the context of CNNs and explain, using a concrete example, why it leads to translation invariance.

3. What is a receptive field? Explain how the receptive field of a neuron in layer 3 of a CNN is larger than that of a neuron in layer 1, and why this matters for the type of features each neuron can detect.

4. Draw a simple ASCII diagram showing the data flow through a CNN from a raw 32×32 colour image to a 10-class probability output. Label each layer with its type and approximate output shape.

5. Consider a task of classifying 1-second audio clips by genre. Would a dense network or a CNN architecture be more appropriate? Explain your reasoning by drawing parallels between audio spectrograms and images.

---

## Further Reading

- **"An Intuitive Explanation of Convolutional Neural Networks" by Ujjwal Karn** — Freely available blog post with excellent visualisations of parameter sharing and feature maps; a great complement to this lesson.
- **cs231n.stanford.edu: Convolutional Neural Networks lecture notes** — Stanford's authoritative explanation of CNNs with detailed diagrams, covering everything in this lesson and more.
- **"Visualizing and Understanding Convolutional Networks" (Zeiler & Fergus, 2013)** — The paper that first visualised what CNN filters learn, proving that early layers detect edges and later layers detect complex objects.
- **Keras documentation: Conv2D layer** — Technical documentation for the Conv2D layer including all parameters, input/output shapes, and usage examples.
- **"Deep Learning for Computer Vision" (Francois Chollet, Chapter 5)** — The author of Keras provides a hands-on explanation of why dense networks fail for images and how CNNs solve the problem.
