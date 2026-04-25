# Lesson 15: Batch Normalization and Dropout

## Learning Objectives
- Explain the internal covariate shift problem and why it makes training deep networks difficult
- Describe the batch normalization algorithm step-by-step, including the learnable scale and shift parameters
- Compare placing BatchNorm before vs. after activation functions and understand the ongoing debate
- Explain the difference between BatchNorm behavior during training vs. inference
- Describe the dropout mechanism and justify dropout rate selection for different layer types
- Analyze how BatchNorm and Dropout interact and when to use each or both

---

## Detailed Explanation

### The Deep Network Training Problem

When training a deep neural network, a subtle but important problem arises: as each layer's weights update during training, the distribution of inputs to every subsequent layer keeps changing. Layer 3 trains assuming a certain distribution from Layer 2, but Layer 2's weights are also changing — so Layer 3 is essentially chasing a moving target. This phenomenon is called **internal covariate shift**.

Imagine trying to aim a cannon at a target, but the ground under the cannon shifts slightly after every shot. You adjust your aim, but then the ground shifts again. Progress is slow and unstable. This is exactly what deep networks experience: each layer must constantly re-adapt to the changing output distribution of the previous layer.

Internal covariate shift causes:
- The need for very small learning rates (large steps cause instability)
- Slow convergence
- Vanishing or exploding gradients in very deep networks
- Sensitivity to weight initialization

**Batch Normalization**, introduced by Ioffe and Szegedy in 2015, directly addresses this by normalizing layer inputs to have zero mean and unit variance at every mini-batch.

---

### How Batch Normalization Works

Given a mini-batch of activations `{x_1, x_2, ..., x_m}` for a single neuron (or channel, for CNNs):

**Step 1: Compute batch mean**
```
μ_B = (1/m) * Σ x_i
```

**Step 2: Compute batch variance**
```
σ²_B = (1/m) * Σ (x_i - μ_B)²
```

**Step 3: Normalize**
```
x̂_i = (x_i - μ_B) / sqrt(σ²_B + ε)
```
The small constant ε (typically 1e-5) prevents division by zero.

**Step 4: Scale and Shift with learnable parameters γ and β**
```
y_i = γ * x̂_i + β
```

The learnable parameters γ (scale) and β (shift) are what make BatchNorm different from simple standardization. Without them, the normalization would constrain every layer's output to mean 0, std 1 — which might be suboptimal. γ and β allow the network to learn the optimal scale and shift for each feature, potentially even learning to "undo" the normalization if that is what the task requires.

This is a crucial insight: BatchNorm doesn't force a fixed distribution. It provides a differentiable way to control the distribution, giving the optimizer the choice of what distribution works best for each feature.

```
Mini-batch values:  [1.2, 3.5, 2.1, 4.8, 0.9]
After BatchNorm:    [-0.9, 0.6, -0.2, 1.3, -1.1]  ← zero mean, unit variance
After scale+shift:  [-1.35, 0.9, -0.3, 1.95, -1.65] ← γ=1.5, β=0
```

---

### Why BatchNorm Dramatically Improves Training

1. **Allows higher learning rates**: Normalized inputs make the loss landscape more "bowl-shaped" and smooth, allowing larger gradient steps without divergence.

2. **Reduces sensitivity to initialization**: Before BatchNorm, careful weight initialization was critical. After BatchNorm, initialization matters much less because the activations are re-normalized at every layer.

3. **Acts as a regularizer**: The noise introduced by computing statistics on mini-batches (rather than the full dataset) adds stochastic variation to activations. This acts similarly to dropout and often reduces the need for explicit dropout.

4. **Faster convergence**: In practice, networks with BatchNorm often converge 10x faster than those without it. What took 100 epochs might take only 10 epochs.

---

### Where to Place BatchNorm: Before or After Activation?

This is a genuinely debated question in the deep learning community.

**Original paper placement** (Ioffe & Szegedy, 2015): BatchNorm between the linear transformation and the activation:
```
Linear → BatchNorm → Activation
(Dense or Conv) → BN → ReLU
```

**Alternative placement** (many modern practitioners): BatchNorm AFTER the activation:
```
Linear → Activation → BatchNorm
(Dense or Conv) → ReLU → BN
```

The original placement is more theoretically motivated (normalize inputs to the activation function so it operates in its sensitive linear region). The alternative placement avoids the interaction between ReLU (which zeros negative values) and normalization (which might then shift the mean non-trivially).

In practice, both approaches work. The original (BN before activation) is still more common. For residual networks (ResNets), BN before activation is standard. Experiment with your specific architecture and dataset.

---

### BatchNorm During Training vs. Inference

During **training**, BatchNorm uses the statistics (mean and variance) of the current mini-batch. This introduces useful noise.

During **inference**, using mini-batch statistics would make predictions depend on which other samples happen to be in the batch — unacceptable for deployment. Instead, BatchNorm uses **running averages** accumulated during training:

```
running_mean = momentum * running_mean + (1 - momentum) * batch_mean
running_var  = momentum * running_var  + (1 - momentum) * batch_var
```

Keras handles this automatically via the `training=True/False` flag, which is set by `model.fit()` vs. `model.predict()`. When you call `model(inputs, training=False)` (inference mode), BatchNorm uses the running statistics.

**Common Bug**: If you call `model(inputs)` without explicitly setting `training=False` in a custom training loop, BatchNorm might continue using batch statistics during evaluation, leading to misleadingly good validation metrics.

---

### Dropout: A Complementary Technique

While BatchNorm addresses training stability and speed, **Dropout** addresses overfitting by randomly deactivating neurons during training.

At each training step, each neuron in a dropout layer is independently set to zero with probability `p` (the dropout rate) and kept with probability `(1-p)`. The remaining active neurons are scaled by `1/(1-p)` to maintain the expected sum.

```
Dropout Process (p=0.5):
Input:     [0.8,  0.3,  0.6,  0.9,  0.2]
Mask:      [1,    0,    1,    0,    1  ]   ← random binary mask
Output:    [1.6,  0.0,  1.2,  0.0,  0.4]  ← kept values scaled by 1/(1-0.5)=2
```

**Why scaling?** Without scaling, the expected sum of outputs would change between training (partial neurons) and inference (all neurons). Scaling ensures the expected activation magnitude remains consistent regardless of the dropout rate.

---

### Choosing Dropout Rates

| Layer Type | Typical Dropout Rate | Reasoning |
|------------|---------------------|-----------|
| Large Dense layers | 0.4 – 0.5 | Dense layers have many weights; aggressive dropout needed |
| Small Dense layers | 0.2 – 0.3 | Less capacity to spare |
| CNN Conv layers | 0.1 – 0.25 | Spatial correlation means standard dropout is less effective |
| After pooling layers | 0.2 – 0.4 | Good regularization point |
| Output layer | 0 (never) | Never apply dropout to the final classification layer |

**Spatial Dropout2D**: For convolutional layers, drops entire feature maps (channels) rather than individual neurons. Because neighboring pixels in a feature map share spatial context, dropping individual pixels is inefficient — whole channels carry more semantically coherent information.

---

### BatchNorm and Dropout Interaction

Using BatchNorm and Dropout together requires care. When Dropout is applied before BatchNorm, it changes the effective mean and variance seen by BatchNorm, because dropped-out activations are zero. This creates a training/inference discrepancy: during training, many zeros are included in the batch statistics; during inference, no zeros are present.

**Best practice** (supported by research by Li et al., 2019 "Understanding the Disharmony between Dropout and Batch Normalization"):
- In networks with BatchNorm in convolutional layers, use **Spatial Dropout** in those layers
- Apply Dropout AFTER BatchNorm layers (not before) if both are used in the same block
- Consider using only BatchNorm in convolutional blocks and only Dropout in fully connected blocks
- In modern architectures (ResNets, EfficientNets), BatchNorm alone is often sufficient; Dropout is added only to the final dense layers

---

### ASCII Diagram: CNN Block Comparison

```
WITHOUT BN/Dropout:           WITH BN + Dropout:
┌────────────────┐            ┌────────────────┐
│  Conv2D        │            │  Conv2D        │
├────────────────┤            ├────────────────┤
│  ReLU          │            │  BatchNorm     │
├────────────────┤            ├────────────────┤
│  Conv2D        │            │  ReLU          │
├────────────────┤            ├────────────────┤
│  ReLU          │            │  Conv2D        │
├────────────────┤            ├────────────────┤
│  Dense         │            │  BatchNorm     │
├────────────────┤            ├────────────────┤
│  Softmax       │            │  ReLU          │
└────────────────┘            ├────────────────┤
                              │  Dense         │
                              ├────────────────┤
                              │  Dropout(0.4)  │
                              ├────────────────┤
                              │  Softmax       │
                              └────────────────┘
```

---

### Common Misconceptions

1. **"BatchNorm and Dropout always improve performance"** — Not always. For very small models or small datasets, they may introduce too much regularization. Always validate on your specific task.
2. **"BatchNorm is just normalization"** — The learnable γ and β are what make it a generalization of normalization. The network can learn to amplify or shift features beyond the normalized range.
3. **"Dropout at inference time is just turned off"** — Correct behavior, but a common bug in custom inference code. Always ensure `training=False` is set during evaluation.
4. **"Large dropout rates are always better for regularization"** — Too high a dropout rate causes underfitting by preventing the network from learning stable representations.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Internal Covariate Shift | Distribution of layer inputs changing during training as weights update | Root cause of instability in deep networks |
| Batch Normalization | Normalizes activations across batch dimension, then scales/shifts | Enables faster training, higher LR, better generalization |
| γ (gamma) | Learnable scale parameter in BatchNorm | Allows network to control activation magnitude |
| β (beta) | Learnable shift parameter in BatchNorm | Allows network to control activation mean |
| Running Statistics | Exponential moving averages of batch mean/var accumulated during training | Used at inference time for stable, batch-independent predictions |
| Dropout | Randomly zeroes neurons during training with probability p | Ensemble-like regularization; prevents co-adaptation |
| Dropout Rate | Fraction of neurons dropped (probability p) | Controls strength of regularization |
| Spatial Dropout2D | Drops entire feature map channels in CNNs | More effective than neuron-level dropout for convolution layers |
| Training Mode | Model state during fit; BatchNorm uses batch stats, Dropout is active | Phase where weights are updated and stochasticity is present |
| Inference Mode | Model state during predict; BatchNorm uses running stats, Dropout off | Phase for stable, deterministic predictions |

---

## Code Reference

See the full runnable demo in **code/lesson_15.py**

---

## Activities

1. **BatchNorm Training Speed**: Build two identical CNNs for CIFAR-10 — one with BatchNorm after each Conv layer and one without. Train both for 20 epochs. Compare training loss curves and time to reach 65% validation accuracy. How many fewer epochs does BatchNorm require?

2. **Learning Rate Sensitivity Test**: Using a model without BatchNorm, try learning rates {0.1, 0.01, 0.001}. Then add BatchNorm and repeat. Show in a 2×3 grid of training curves how BatchNorm makes the model tolerant to higher learning rates.

3. **Dropout Rate Exploration**: Build a Dense network for MNIST with one hidden layer of 512 units. Add a Dropout layer with rates {0.0, 0.2, 0.4, 0.6, 0.8}. Plot training and validation accuracy for all rates on the same graph. Identify the rate that achieves the best validation accuracy.

4. **Inference Mode Bug**: In a model with Dropout and BatchNorm, compute accuracy in two ways: (a) using `model.evaluate()` and (b) manually calling `model(X_test)` without setting `training=False`. Observe and explain any accuracy difference. Fix the bug.

5. **BatchNorm + Dropout Interaction**: Implement two variants of a CNN: (a) BatchNorm in conv blocks + Dropout before the final dense layer, (b) No BatchNorm + Dropout in conv blocks. Compare validation accuracy and training stability over 20 epochs.

---

## Review Questions

1. What is internal covariate shift and why does it slow down training? How does Batch Normalization address it?
2. Explain the role of the learnable parameters γ and β in BatchNorm. Why are they necessary? What would happen if we removed them?
3. Why does BatchNorm behave differently during training vs. inference? What bug can arise if inference mode is not properly set?
4. Describe the dropout mechanism step by step. Why must the remaining neurons be scaled during training? What happens at inference time?
5. Why might using Dropout before BatchNorm cause problems? What placement of these two layers is recommended and why?

---

## Further Reading

- "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" — Ioffe & Szegedy, 2015 (arXiv:1502.03167)
- "Dropout: A Simple Way to Prevent Neural Networks from Overfitting" — Srivastava et al., JMLR 2014
- "Understanding the Disharmony between Dropout and Batch Normalization" — Li et al., CVPR 2019
- "Layer Normalization" — Ba et al., 2016 — an alternative to BatchNorm for sequence models
- Keras documentation: `tf.keras.layers.BatchNormalization` and `tf.keras.layers.Dropout`
