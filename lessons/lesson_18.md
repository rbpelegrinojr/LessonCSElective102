# Lesson 18: Model Visualization & Interpretability (Grad-CAM)

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why interpretability is essential for real-world deep learning systems
- Describe saliency maps and guided backpropagation as gradient-based explanation methods
- Understand Class Activation Maps (CAM) and their dependency on Global Average Pooling
- Implement Grad-CAM from scratch using `tf.GradientTape`
- Generate and overlay Grad-CAM heatmaps on original images
- Use intermediate layer activations to visualize what different parts of a CNN have learned
- Compare Grad-CAM outputs for correct and incorrect predictions to debug model behavior

---

## Detailed Explanation

### Why Interpretability Matters

Modern deep neural networks are extraordinarily powerful, but they operate as **black boxes** — given an input, they produce an output, with little transparency about what internal process drove that decision. For many real-world applications, this opacity is not acceptable:

- **Medical imaging**: A radiologist needs to know *which part of the X-ray* the model flagged as abnormal before trusting its diagnosis.
- **Autonomous driving**: A safety engineer must verify that the system is detecting the actual road markings and not spurious artifacts.
- **Regulatory compliance**: In the EU and other regions, AI systems used in high-stakes decisions must be explainable to affected individuals (e.g., GDPR's "right to explanation").
- **Debugging**: If your model has low accuracy on a particular class, visualization can reveal whether it is looking at irrelevant parts of the image (background color, watermarks, image borders).
- **Trust building**: Stakeholders and clients are more willing to deploy a model when they can see that it is reasoning correctly.

### The Black Box Problem

A standard CNN with millions of parameters transforms an input image through dozens of nonlinear operations before producing a class probability. No single weight or layer is responsible for the final prediction — the answer emerges from the complex, distributed computation of the entire network. This makes it nearly impossible to inspect the network by examining weights directly.

Instead, **post-hoc explanation methods** probe the trained network from the outside, analyzing how outputs change in response to input perturbations or gradients.

### Saliency Maps

The simplest explanation method is the **saliency map** (Simonyan et al., 2013). The idea is: compute the gradient of the class score with respect to the input image. Pixels where a small change causes a large change in the class score are "important" to the prediction.

```
saliency = ∂(class_score) / ∂(input_image)
```

This gradient is computed using backpropagation, and its magnitude at each pixel indicates that pixel's influence on the final prediction. The result is a map of the same spatial dimensions as the input, showing which pixels matter most.

**Limitations**: Saliency maps can be noisy and sensitive to irrelevant local variations. They do not distinguish between positive and negative evidence.

### Guided Backpropagation

**Guided backpropagation** (Springenberg et al., 2014) improves on saliency maps by modifying the backpropagation rule: during the backward pass through ReLU activations, gradients are only allowed to flow back through neurons that were **both positive in the forward pass AND have positive gradients**. This produces crisper, more interpretable visualizations that highlight the actual features driving the prediction rather than uninformative noise.

### Class Activation Maps (CAM)

**CAM** (Zhou et al., 2015) was the breakthrough that made spatial localization from classification models possible. The key insight is that the final convolutional layer's feature maps retain spatial information — each feature map channel is a 2D spatial representation of what a particular filter detects across the image.

In a network that ends with **GlobalAveragePooling2D** (GAP), the output is a weighted average of the spatial feature maps. The weights learned in the final Dense layer tell us how important each feature map channel is for predicting each class. Therefore:

```
CAM_c = Σ_k  w_k^c  *  A_k
```

where `w_k^c` is the weight for class `c` and feature map `k`, and `A_k` is the spatial feature map. Upsampling this to the original image size gives a heatmap highlighting the most discriminative regions.

**Limitation**: CAM requires a specific architecture — the model must use GlobalAveragePooling directly before the classification Dense layer. It cannot be applied to arbitrary architectures.

### Grad-CAM: The General Solution

**Grad-CAM** (Selvaraju et al., 2017) generalizes CAM to **any convolutional network architecture**, regardless of whether it uses GAP. Instead of using the Dense layer weights, Grad-CAM uses **gradients** to determine the importance of each feature map channel.

**How Grad-CAM works, step by step:**

1. **Forward pass**: Run the image through the network. Record the activations of the target convolutional layer (usually the last conv block).
2. **Compute gradient**: Backpropagate the gradient of the class score (before softmax) with respect to the recorded feature maps.
3. **Global average pooling of gradients**: For each feature map channel `k`, average the gradient values across all spatial positions. This gives a scalar weight `α_k^c` for each channel:

```
α_k^c = (1/Z) * Σ_i Σ_j  ∂y^c / ∂A^k_{ij}
```

4. **Weighted sum of feature maps**: Compute a weighted combination of the feature maps, using the channel importances as weights:

```
L^c_Grad-CAM = ReLU( Σ_k  α_k^c  *  A_k )
```

5. **Apply ReLU**: The ReLU operation keeps only the features that have a **positive** effect on the class score (features that increase the probability of class `c`). Negative-contributing regions are discarded.

6. **Upsample and overlay**: The resulting low-resolution heatmap is upsampled (using bilinear interpolation) to the size of the input image and overlaid as a semi-transparent color map.

```
ASCII Diagram: Grad-CAM Computation Flow

Input Image
     │
     ▼
┌──────────────────────────────────────────┐
│         Convolutional Blocks             │
│  Conv1 → Conv2 → ... → Conv_target (A_k)│◄── Record activations here
└──────────────────────────────────────────┘
     │                        │
     ▼                        │ Backprop gradients
┌─────────┐            ┌─────────────────────┐
│ GAP/    │            │  ∂y^c / ∂A^k_{ij}   │
│Flatten  │            │  (gradient maps)    │
└────┬────┘            └──────────┬──────────┘
     │                            │
     ▼                            ▼
┌─────────┐            ┌──────────────────────┐
│  Dense  │            │  Global Average Pool │
│(Softmax)│            │  → α_k^c (weights)   │
└────┬────┘            └──────────┬───────────┘
     │                            │
     │                 ┌──────────▼───────────┐
     │                 │  Σ α_k^c * A_k       │
     │                 │  → ReLU → Heatmap    │
     │                 └──────────┬───────────┘
     │                            │
     │                 ┌──────────▼───────────┐
     │                 │  Upsample to input   │
     │                 │  size, overlay on    │
     │                 │  original image      │
     ▼                 └──────────────────────┘
Class Prediction
```

### Grad-CAM++ for Better Localization

**Grad-CAM++** (Chattopadhay et al., 2018) is an improvement over Grad-CAM that provides better localization when multiple instances of the same object appear in the image, or when the object of interest is only partially visible. It uses a more sophisticated weighting formula for the gradients that accounts for higher-order derivatives. For most single-object classification tasks, standard Grad-CAM is sufficient.

### Overlaying the Heatmap

To create the final visualization, the Grad-CAM heatmap is overlaid on the original image:

1. Normalize the heatmap to [0, 1]
2. Apply a colormap (e.g., `jet` or `viridis`) to convert to RGB
3. Resize to match the input image
4. Blend with the original image using alpha transparency

The resulting image shows hot regions (red/yellow) where the model is "looking" for the predicted class.

### Applications of Grad-CAM

- **Verify correctness**: Confirm the model focuses on the object (e.g., the bird's beak) rather than irrelevant background features
- **Failure analysis**: When the model makes an incorrect prediction, Grad-CAM often reveals it is looking at the wrong part of the image or at spurious correlations
- **Dataset bias detection**: If a "horse classifier" highlights the caption/watermark rather than the horse, your dataset has a bias problem
- **Medical AI**: Radiologists can overlay heatmaps on X-rays or MRIs to see which anatomical regions drive the AI's decision

### Intermediate Layer Visualization

Beyond Grad-CAM, you can directly visualize what each layer of the CNN has learned by creating a **feature map model** — a Keras model that outputs the activations of any intermediate layer:

```python
layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name]
activation_model = tf.keras.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(sample_image)
```

Plotting these activation maps shows that early layers respond to edges and colors, while later layers respond to increasingly abstract and task-specific patterns.

---

## Key Concepts Table

| Concept | Definition |
|---|---|
| Interpretability | The ability to explain why a model made a particular prediction |
| Saliency Map | Gradient of class score w.r.t. input pixels; highlights important pixels |
| Guided Backpropagation | Modified backprop that suppresses noise for sharper pixel-level attribution |
| CAM | Class Activation Map; requires GAP architecture; spatially localizes discriminative regions |
| Grad-CAM | Generalization of CAM using gradients; works on any CNN architecture |
| GradientTape | TensorFlow API to record operations for automatic differentiation |
| Heatmap Overlay | Blending the Grad-CAM heatmap with the original image for visualization |
| Feature Map | The 2D spatial output of a convolutional layer for a given input |
| Activation Model | A Keras model that outputs intermediate layer activations |
| Failure Analysis | Using Grad-CAM to understand why the model made wrong predictions |

---

## Code Reference

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

def grad_cam(model, image, layer_name, class_index):
    """Compute Grad-CAM heatmap for a given image and target class."""
    # Create a sub-model that outputs both the target conv layer and predictions
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(tf.expand_dims(image, 0))
        loss = predictions[:, class_index]  # Score for the target class

    # Gradient of the class score w.r.t. the conv layer output
    grads = tape.gradient(loss, conv_outputs)

    # Global average pooling of gradients → importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight each feature map by its importance
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()
```

---

## Activities

1. **Visualize Feature Maps**: Pick a trained CNN. Create an activation model and visualize the feature maps of the first, middle, and last convolutional layers on a sample image. What patterns do you see?

2. **Grad-CAM on Correct Predictions**: Apply Grad-CAM to 5 correctly classified images. Does the heatmap focus on the relevant part of the image?

3. **Grad-CAM on Incorrect Predictions**: Find 5 misclassified images. Apply Grad-CAM. Can you explain why the model got it wrong based on where it was "looking"?

4. **Compare Layers**: Apply Grad-CAM using different convolutional layers (first conv vs. last conv) as the target layer. How do the heatmaps differ?

5. **Filter Maximization**: For the first convolutional layer, generate the input image that maximally activates a specific filter by gradient ascent. What pattern does it prefer?

---

## Review Questions

1. Why is interpretability important for deep learning models used in medical applications?
2. How does a saliency map work, and what is its main limitation?
3. What architectural requirement does the original CAM method have?
4. Explain the three main steps of Grad-CAM in your own words.
5. Why is a ReLU applied to the weighted sum of feature maps in Grad-CAM?
6. How would you use Grad-CAM to detect dataset bias?
7. What is `tf.GradientTape` and why is it needed for Grad-CAM?
8. What is the difference between Grad-CAM and Grad-CAM++?

---

## Further Reading

- [Grad-CAM: Visual Explanations from Deep Networks (Selvaraju et al., 2017)](https://arxiv.org/abs/1610.02391)
- [Class Activation Mapping (Zhou et al., 2015)](https://arxiv.org/abs/1512.04150)
- [Guided Backpropagation (Springenberg et al., 2014)](https://arxiv.org/abs/1412.6806)
- [tf-explain: Interpretability for TensorFlow](https://github.com/sicara/tf-explain)
- [SHAP for Deep Learning](https://github.com/slundberg/shap)
- [Interpretable Machine Learning – Christoph Molnar (free book)](https://christophm.github.io/interpretable-ml-book/)
