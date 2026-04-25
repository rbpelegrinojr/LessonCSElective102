# Lesson 18: Model Visualization and Interpretability with Grad-CAM

## Learning Objectives
- Explain why model interpretability matters for trust, debugging, and regulatory compliance in AI systems
- Describe what feature maps and convolutional filters represent at different depths of a CNN
- Implement Grad-CAM (Gradient-weighted Class Activation Mapping) from scratch using TensorFlow's `GradientTape`
- Visualise feature maps and activation heatmaps and interpret what regions of an image drive predictions
- Distinguish between Grad-CAM, saliency maps, and occlusion sensitivity as interpretability approaches
- Identify failure modes and biases in a trained CNN using visualisation tools

## Detailed Explanation

### Why Interpretability Matters

A model that makes correct predictions is useful. A model whose reasoning you can inspect and verify is trustworthy. Consider these scenarios:

- A CNN classifying chest X-rays achieves 94 % accuracy — but is it looking at the lung tissue, or is it exploiting a hospital watermark that happens to correlate with disease severity in the training set?
- An autonomous vehicle's obstacle detector misclassifies a stop sign — was it because the sign was partially occluded, or because a small adversarial sticker was placed on it?
- A credit-risk model built on financial data must, by law in the EU, provide an explanation for every automated decision.

Visualisation and interpretability tools let us answer the question: **What is the model actually looking at when it makes this prediction?** This bridges the gap between a black-box accuracy number and genuine understanding of model behaviour.

### What Lives Inside a CNN?

#### Layer 1: Edge Detectors

The very first convolutional layer typically learns a set of low-level filters that respond to oriented edges, colour oppositions, and local frequency patterns. If you visualise the 64 filters of the first conv layer of VGG16, you will see a collection of horizontal-edge detectors, vertical-edge detectors, diagonal-edge detectors, and colour-difference filters — almost identical to what we find in the primary visual cortex (V1) of mammals.

#### Intermediate Layers: Texture and Part Detectors

As you go deeper, the receptive field grows and the patterns become more complex. Middle layers respond to textures (fur, scales, fabric), repeated motifs (grids, stripes), and simple shapes (arcs, blobs). These intermediate representations are what make features transferable: textures are useful across many domains.

#### Final Conv Layers: Semantic Detectors

The last few convolutional layers contain filters that respond to high-level semantic patterns — dog faces, wheel shapes, eye patterns. These are the layers whose feature maps carry the richest information about which class is being detected. This is also why Grad-CAM focusses on the last convolutional layer.

### Activation Maps (Feature Maps)

When an image passes through a CNN, every convolutional layer produces a volume of **feature maps** — one spatial map per filter. Each map is a 2D grid showing where in the image that filter activated strongly.

```
Input: 224×224×3 image  (H × W × Channels)
       │
       ▼
Conv Layer 1:  224×224×64   — 64 feature maps, each 224×224
       │
       ▼
After MaxPool: 112×112×64
       │
       ▼
Conv Layer 2:  112×112×128  — 128 feature maps, each 112×112
       │
       ▼
  ...continues...
       │
       ▼
Last Conv:     7×7×512      — 512 feature maps, each 7×7
```

Plotting these feature maps as greyscale images reveals what each filter "sees" at that layer. Early filters produce sharp, edge-like activations; deeper filters produce diffuse, blob-like activations that correspond to object parts.

### Class Activation Maps (CAM)

**CAM** (Zhou et al., 2016) was one of the first methods to produce visual explanations of CNN predictions without modifying the architecture. It works for networks with a **Global Average Pooling** layer just before the final classification layer:

```
Last Conv Layer  →  GlobalAveragePooling  →  Dense (softmax)
    ↓                       ↓                      ↓
7×7×512 maps         512-dim vector          class scores

For class c:
  CAM_c(x, y) = Σ_k  w_k^c  ×  f_k(x, y)
```

Where `w_k^c` are the weights from the dense layer connecting feature map `k` to class `c`, and `f_k(x,y)` is the activation of map `k` at spatial position `(x,y)`. The result is a single 7×7 heatmap that, when upscaled to the original image size, shows which spatial regions were most important for predicting class `c`.

**Limitation of CAM:** It requires the specific architecture (GAP → Dense). It cannot be applied to networks with fully connected layers between the last conv and the output.

### Grad-CAM: The General Solution

**Grad-CAM** (Selvaraju et al., 2017) generalises CAM to work with any CNN architecture by replacing the fixed classification weights with gradients:

#### Grad-CAM Algorithm, Step by Step

```
Step 1: Forward pass the image to get the predicted class score y^c

Step 2: Backpropagate y^c to the last convolutional layer A^k
        (do NOT update weights — just compute gradients)

Step 3: Compute importance weights for each feature map k:
        α_k^c = (1 / Z) × Σ_{i,j}  ∂y^c / ∂A^k_{ij}
        (global average pool of the gradients)

Step 4: Form the weighted combination:
        L^c_Grad-CAM = ReLU( Σ_k  α_k^c × A^k )
        (ReLU discards negative contributions — we want activations
         that INCREASE the class score, not decrease it)

Step 5: Upsample L^c_Grad-CAM (7×7) to input image size (224×224)
        using bilinear interpolation

Step 6: Overlay as a heatmap (jet or inferno colormap) on original image
```

The **ReLU** in Step 4 is critical: without it, regions that strongly vote *against* the class would appear as negative values and confuse the visualisation. We care only about what evidence *supports* the class prediction.

#### Visual Intuition

```
Input Image         Last Conv Feature Maps    Grad-CAM Heatmap    Overlay
┌───────────┐       ┌─┬─┬─┬─┬─┐             ┌───────────┐       ┌───────────┐
│           │  →    │ │ │ │ │ │  → weights → │ hot: red  │  →    │ ████      │
│  🐱 cat   │       │ │ │ │ │ │             │ cool: blue│       │ ██ face ██│
│           │       └─┴─┴─┴─┴─┘             └───────────┘       └───────────┘
                    (7×7 × 512)                (7×7 → 224×224)
```

Red regions in the heatmap correspond to areas the model focussed on most heavily when predicting "cat". If those regions are the cat's face and body, the model is behaving correctly. If they are the background, the model may be exploiting spurious correlations.

### Saliency Maps

**Saliency maps** compute the gradient of the output class score with respect to the input pixels. Large gradient magnitude at pixel (i,j) means that small changes to that pixel strongly affect the prediction — so the model is sensitive to it.

```
Saliency(i,j) = |∂y^c / ∂x_{i,j}|
```

Saliency maps are fast and simple but tend to produce noisy, scattered visualisations that are hard to interpret. Grad-CAM produces much smoother, more spatially coherent heatmaps.

### Occlusion Sensitivity

A simple, model-agnostic interpretability technique: systematically occlude small patches of the image with a grey square and record how much the prediction probability drops.

```
For each position (i,j):
  1. Mask a k×k region centred on (i,j) with grey
  2. Run the masked image through the model
  3. Record drop in probability for the predicted class

Large drop → that region was important
Small drop → that region was not important
```

Occlusion sensitivity is intuitive and faithful to the model's actual behaviour, but it is computationally expensive (requires a forward pass per patch).

### What Grad-CAM Tells Us About Model Decisions

| Observation | Interpretation | Action |
|---|---|---|
| Heatmap covers the correct object | Model is using relevant features | Good — model is trustworthy |
| Heatmap is on the background | Model exploits background correlations | Augment with varied backgrounds; re-evaluate training data |
| Heatmap is scattered/diffuse | Model is uncertain or confused | Investigate difficult examples; may need more data |
| Heatmap shifts for wrong prediction | Model confuses similar objects | Add hard negatives; consider focal loss |
| Heatmap covers metadata (watermark, ruler) | Model uses spurious correlations | Remove metadata from images; re-train |

### Using Interpretability to Debug Models

Grad-CAM can reveal serious problems before deployment:

- **Clever Hans Effect**: A famous AI scandal where a horse classifier achieved near-perfect accuracy by detecting the copyright watermark present in most horse images in the training set, not the horse itself. Grad-CAM would immediately expose this.
- **Shortcut Learning**: Models often learn the easiest discriminating feature, not the most meaningful one. Grad-CAM surfaces these shortcuts.
- **Dataset Bias**: If all "doctor" images show male faces and all "nurse" images show female faces, the model may classify by gender instead of role.

### LIME for Images

**LIME** (Locally Interpretable Model-agnostic Explanations) is a model-agnostic approach that explains any black-box classifier by:
1. Segmenting the image into superpixels
2. Randomly switching superpixels on/off to create perturbed versions
3. Running all perturbed versions through the model
4. Fitting a simple linear model to the perturbation results
5. The coefficients reveal which superpixels most positively influenced the prediction

LIME does not require access to gradients, making it applicable to any model (including sklearn, XGBoost, etc.), but it is slow compared to gradient-based methods.

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| Interpretability | The degree to which humans can understand the cause of a model's decision | Essential for trust, debugging, and regulatory compliance |
| Feature Map | The 2D activation output of a single filter applied to an input | Reveals what spatial patterns a filter detects at each layer |
| Grad-CAM | Uses backpropagated gradients to weight feature maps and highlight important regions | Works with any CNN; produces smooth, class-specific heatmaps |
| Saliency Map | Gradient of the output w.r.t. input pixels; shows pixel-level sensitivity | Fast but noisy; coarser than Grad-CAM |
| CAM | Class Activation Map using weights of the final linear layer | Predecessor to Grad-CAM; requires GAP architecture |
| Occlusion Sensitivity | Masking image patches to measure their importance | Intuitive and model-agnostic but computationally expensive |
| GradientTape | TensorFlow mechanism for recording and computing gradients manually | Essential for implementing Grad-CAM from scratch |
| Receptive Field | The region of the input image that influences a neuron's activation | Grows with depth; last conv layer neurons see most of the image |
| Clever Hans Effect | Model appears to succeed but uses irrelevant spurious features | Shows why accuracy alone is insufficient; visualisation reveals the truth |
| LIME | Model-agnostic explanation by fitting a local linear model to perturbed inputs | Applicable to any classifier; good complement to gradient methods |

## Code Reference

See the fully runnable demonstration in **`code/lesson_18.py`**.

## Activities

1. **Feature Map Gallery** — Train a small CNN on CIFAR-10. Extract and plot the first 16 feature maps from layer 1, layer 2, and the last conv layer for a single test image. Describe in writing what visual patterns each layer appears to detect.

2. **Grad-CAM from Scratch** — Following the algorithm in this lesson, implement Grad-CAM using `tf.GradientTape` on a pretrained MobileNetV2. Generate heatmaps for five different CIFAR-10 test images. Overlay the heatmap on the original image using a 50 % transparency blend.

3. **Class Comparison** — For the same input image, generate Grad-CAM heatmaps for three different output classes (e.g. the predicted class, the second-best class, and a completely wrong class). Display them side by side and explain the differences.

4. **Saliency Map vs Grad-CAM** — Implement a simple saliency map (gradient of class score w.r.t. input image) for the same five test images. Display saliency maps and Grad-CAM heatmaps side by side. Which is more interpretable? Write a two-sentence comparison.

5. **Occlusion Sensitivity** — Implement a 16×16 occlusion patch that slides across a 64×64 image with stride 8. For each position, record the drop in prediction confidence. Plot the result as a 2D heatmap. Compare it to the Grad-CAM for the same image.

## Review Questions

1. Explain the Grad-CAM algorithm step by step, including why the ReLU is applied to the final weighted combination.
2. Why might a model that achieves high test accuracy still be untrustworthy? Give a real-world example where interpretability would be critical.
3. Compare Grad-CAM and saliency maps: what does each compute, and what are the practical advantages of Grad-CAM for generating explanations?
4. A Grad-CAM heatmap for a "tumour present" prediction shows activation concentrated on the image metadata (date stamp) rather than the scan tissue. What does this tell you, and what would you do?
5. Explain the Clever Hans Effect in the context of machine learning. How can Grad-CAM help detect it?

## Further Reading

- **"Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"** — Selvaraju et al. (2017) — the original Grad-CAM paper with full algorithm derivation
- **"Learning Deep Features for Discriminative Localization"** — Zhou et al. (2016) — the original CAM paper, prerequisite reading before Grad-CAM
- **"LIME: Why Should I Trust You?"** — Ribeiro et al. (2016) — introduces LIME; highly accessible paper worth reading before production deployment
- **"Visualizing and Understanding Convolutional Networks"** — Zeiler & Fergus (2013) — foundational paper on visualising CNN internals with deconvolution
- **TensorFlow Tutorials: Integrated Gradients** — `https://www.tensorflow.org/tutorials/interpretability/integrated_gradients` — official walkthrough of attribution methods with code
