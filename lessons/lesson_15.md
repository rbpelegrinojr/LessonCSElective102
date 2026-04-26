# Lesson 15: Batch Normalization & Dropout

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain internal covariate shift and why it makes training deep networks difficult
- Describe how batch normalization normalizes activations and what the learnable parameters do
- Place batch normalization correctly in a CNN architecture (before vs after activation)
- Implement and configure Dropout and SpatialDropout2D for regularization
- Combine batch normalization and dropout appropriately in the same model

---

## Detailed Explanation

### The Internal Covariate Shift Problem

Training a deep neural network is challenging because the distribution of each layer's inputs changes during training as the parameters of all preceding layers change. This phenomenon is called **internal covariate shift**, introduced by Ioffe and Szegedy in the 2015 Batch Normalization paper.

Consider a deep network with 20 layers. After each weight update, the outputs of layer 5 change slightly. This means that the inputs to layer 6 have shifted, which changes the inputs to layer 7, and so on. Each layer is constantly trying to adapt to a shifting input distribution. This requires:

- Using very small learning rates (to avoid instability caused by distribution shifts)
- Careful weight initialization (Xavier, He) to keep activations in a reasonable range
- Avoiding saturation in activation functions (tanh/sigmoid neurons saturate when inputs are too large)

The cumulative effect makes training very deep networks slow and sensitive to hyperparameters. Before batch normalization, training networks deeper than ~10 layers was a significant engineering challenge.

### What Batch Normalization Does

**Batch Normalization (BN)** normalizes the activations of each layer across the mini-batch dimension so that they have approximately **zero mean and unit variance**. It is applied to the pre-activation or post-activation values (depending on placement choice) of each channel independently.

For a mini-batch `B = {x₁, x₂, ..., xₘ}`, BN computes:

**Step 1: Compute batch statistics**
```
μ_B = (1/m) * Σ xᵢ          (batch mean)
σ²_B = (1/m) * Σ (xᵢ - μ_B)²  (batch variance)
```

**Step 2: Normalize**
```
x̂ᵢ = (xᵢ - μ_B) / √(σ²_B + ε)
```

Where `ε` (epsilon, typically 1e-5) is a small constant added for numerical stability to prevent division by zero.

**Step 3: Scale and shift with learnable parameters**
```
yᵢ = γ * x̂ᵢ + β
```

Where `γ` (gamma) is the **learnable scale** and `β` (beta) is the **learnable shift**. These parameters are initialized to `γ = 1` and `β = 0` (identity transform) and learned during backpropagation. Their purpose: if the optimal representation for the next layer happens to be non-zero mean or non-unit variance, the network can learn to undo the normalization. This gives BN the flexibility to normalize when helpful but not when it hurts.

### BN During Training vs Inference

**During training:** BN uses the statistics (mean and variance) computed over the current mini-batch.

**During inference:** Computing statistics over a single test example (or a small batch) would be noisy and inconsistent. Instead, BN maintains **running statistics** — exponential moving averages of the batch mean and variance observed during training:

```
μ_running ← momentum * μ_running + (1 - momentum) * μ_B
σ²_running ← momentum * σ²_running + (1 - momentum) * σ²_B
```

> ⚠️ **TensorFlow/Keras convention:** TensorFlow's `BatchNormalization` layer uses `momentum` as the weight for the **old** running value (default `momentum=0.99`), giving:
> `running_stat = 0.99 * running_stat + 0.01 * batch_stat`
> The formula above uses the standard mathematical convention where `momentum` weights the new batch statistic. Both are equivalent with `momentum_keras = 1 - momentum_math`. When using Keras, a higher momentum (e.g., 0.99) means slower adaptation to new batch statistics.

At inference time, the running statistics are used as fixed values. This means BN behaves differently in training mode vs inference mode — a critical implementation detail in Keras (controlled by the `training=True/False` argument).

### Where to Place Batch Normalization

The original BN paper placed normalization **before the activation function**: `Conv → BN → ReLU`. The reasoning: normalize the linear outputs before applying the non-linearity.

However, empirical research and many practitioners have found that placing BN **after the activation** (`Conv → ReLU → BN`) or using specific orderings like `Conv → BN → ReLU` both work well, and results vary by task. The **Conv → BN → ReLU** ordering remains the most widely adopted convention and is what you will see in ResNet and most modern architectures.

In Keras:
```python
tf.keras.layers.Conv2D(64, 3, padding='same', use_bias=False),
tf.keras.layers.BatchNormalization(),
tf.keras.layers.Activation('relu'),
```

Note: when using BN before activation, set `use_bias=False` in the Conv layer — the `β` parameter in BN already provides the bias functionality, so a separate bias term in Conv is redundant.

### Benefits of Batch Normalization

1. **Faster convergence:** Networks with BN can use 5–10× higher learning rates and still converge stably, because the normalized inputs reduce the sensitivity to initialization and learning rate choice.

2. **Reduces sensitivity to initialization:** Since each layer receives normalized inputs, the exact initialization of weights matters less.

3. **Regularization effect:** Because BN computes statistics over a mini-batch, each training example is normalized differently depending on which examples appear in the same batch. This introduces a small amount of noise, which has a mild regularization effect (though not as strong as dropout).

4. **Enables deeper networks:** BN was a critical enabler for training networks with 50+ layers (e.g., ResNet-152 uses BN throughout).

### Dropout: Probability p of Zeroing Neurons

**Dropout** randomly sets neuron outputs to zero during training with probability `p`. The neurons that are kept are scaled by `1/(1-p)` (**inverted dropout**) so that the expected sum of activations remains the same regardless of `p`.

```python
tf.keras.layers.Dropout(rate=0.5)  # Drop 50% of neurons
```

**During inference:** All neurons are active. Because of inverted dropout, no additional scaling is needed. The model effectively performs an approximate ensemble averaging of exponentially many sub-networks.

### SpatialDropout2D for CNNs

Standard Dropout drops individual pixels from feature maps, which is often ineffective for CNNs because neighboring pixels are highly correlated — removing individual pixels still leaves most of the spatial information intact.

**SpatialDropout2D** (also called Feature Map Dropout) drops entire feature maps (channels) at once. If a feature map is dropped, every pixel in that channel becomes zero. This is a more aggressive regularization that forces the model to rely on diverse feature maps rather than a few dominant ones.

```python
tf.keras.layers.SpatialDropout2D(rate=0.2)  # Drop 20% of feature maps
```

Use `SpatialDropout2D` after convolutional layers and standard `Dropout` after fully-connected (Dense) layers.

### Monte Carlo Dropout for Uncertainty Estimation

Normally, dropout is disabled during inference. **Monte Carlo (MC) Dropout** keeps dropout active at inference time and runs the forward pass multiple times (typically 50–100 times) with different random dropout masks. The mean prediction gives a better estimate of the class probability, and the variance across runs gives an estimate of **prediction uncertainty**.

```python
# Run inference with dropout active T times
predictions = [model(X, training=True) for _ in range(T)]
mean_prediction = np.mean(predictions, axis=0)
uncertainty = np.var(predictions, axis=0)
```

High variance across runs means the model is uncertain about that example — useful for anomaly detection and active learning.

### Combining BN and Dropout: Order Matters

When using both BN and Dropout in the same model, the order and placement matter:

**Recommended pattern for convolutional blocks:**
```
Conv2D → BatchNormalization → Activation → SpatialDropout2D
```

**Recommended pattern for fully-connected blocks:**
```
Dense → BatchNormalization → Activation → Dropout
```

A known issue: placing Dropout before BatchNormalization can cause **variance shift**. At training time, Dropout changes the variance of activations that BN then normalizes. At inference time, dropout is off, so the variance is different, but BN still uses training-time running statistics — this mismatch can degrade performance. For this reason, place BN before Dropout.

### ASCII Diagram: Batch Normalization Computation

```
Mini-Batch of m=4 examples, one neuron:

   Input:     [ 2.1,  8.4,  1.2,  5.3 ]
                           ↓
   Batch Mean:  μ_B = (2.1 + 8.4 + 1.2 + 5.3) / 4 = 4.25
   Batch Var:   σ²_B = 7.42
                           ↓
   Normalized:  x̂ = [ -0.79,  1.53, -1.12,  0.38 ]   (mean≈0, std≈1)
                           ↓
   Scaled/Shifted: y = γ * x̂ + β   (γ, β are learned)
```

---

## Key Concepts Table

| Component | Purpose | Training vs Inference |
|---|---|---|
| Batch Normalization | Normalize activations, reduce covariate shift | Uses batch stats (train) / running stats (infer) |
| γ (gamma) | Learnable scale after normalization | Same in both |
| β (beta) | Learnable shift after normalization | Same in both |
| Running Mean/Var | Accumulated statistics for inference | Updated during training, fixed at inference |
| Dropout | Random neuron zeroing, ensemble regularization | Active (train) / Off (infer) |
| SpatialDropout2D | Drop entire feature maps in CNNs | Active (train) / Off (infer) |
| MC Dropout | Dropout at inference for uncertainty | Active in both |
| Inverted Dropout | Scale active neurons by 1/(1-p) during training | No rescaling needed at inference |

---

## Code Reference

```python
import tensorflow as tf

# CNN block with BN and Dropout
def conv_bn_block(filters, kernel_size=3):
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters, kernel_size, padding='same', use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.SpatialDropout2D(0.2),
    ])

# Dense block with BN and Dropout
def dense_bn_block(units, dropout_rate=0.5):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(units, use_bias=False),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(dropout_rate),
    ])
```

---

## Activities

1. **Batch Normalization Stability:** Build two identical 5-layer CNNs for CIFAR-10 — one without any `BatchNormalization` layers and one with a `BatchNormalization` layer after every `Conv2D`. Train both for 10 epochs and plot their training loss curves on the same axes. Comment on the difference in stability.

2. **Dropout Rate Search:** Using the same CNN base, train five models with `Dropout` rates `[0.0, 0.2, 0.4, 0.5, 0.7]` inserted before the final `Dense` layer. After 10 epochs each, plot validation accuracy vs. dropout rate as a line graph and print the rate that gives the highest validation accuracy.
## Review Questions

1. What is internal covariate shift? Why does it make training deep networks difficult?
2. Describe the BN formula. What do γ (gamma) and β (beta) allow the network to do?
3. Why does BN behave differently during training vs inference? What are "running statistics"?
4. Why is `use_bias=False` recommended in Conv layers when followed by BatchNormalization?
5. Why is SpatialDropout2D more appropriate for CNN layers than standard Dropout?
6. Explain inverted dropout. How does it simplify inference code?
7. What is Monte Carlo Dropout, and what additional information does it provide at inference time?
8. When combining BN and Dropout, why should BN come before Dropout?

---

## Further Reading

- Ioffe, S., & Szegedy, C. (2015). *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift*. arXiv:1502.03167
- Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.
- Gal, Y., & Ghahramani, Z. (2016). *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning*. ICML.
- Li, X., et al. (2019). *Understanding the Disharmony Between Dropout and Batch Normalization*. CVPR.
- Keras BatchNormalization documentation: https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
