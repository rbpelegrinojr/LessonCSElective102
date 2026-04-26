# Lesson 11: Loss Functions & Optimizers

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what a loss function is and why it is central to training neural networks
- Distinguish between cross-entropy loss variants (binary, categorical, sparse categorical)
- Describe how gradient descent works and visualize it geometrically
- Compare SGD, Adam, and RMSProp optimizers and choose the right one for a task
- Apply learning rate schedules to improve training stability and convergence

---

## Detailed Explanation

### What Is a Loss Function?

A **loss function** (also called a cost function or objective function) is a mathematical measure of how wrong your model's predictions are compared to the true labels. During training, the neural network's entire goal is to minimize this value. Think of it as the "score" of your model's mistakes — the higher the loss, the worse the predictions.

When the model makes a prediction `ŷ` for a true label `y`, the loss function `L(y, ŷ)` produces a scalar number. The training process adjusts the model's weights to drive this number toward zero (or a small minimum). Choosing the right loss function is critical: the wrong choice can make a perfectly good architecture fail to learn anything useful.

### Cross-Entropy Loss for Classification

The most widely used loss function for classification tasks is **cross-entropy loss**, also called **log loss**. It originates from information theory and measures how different two probability distributions are.

For a single example, categorical cross-entropy is defined as:

```
H(y, ŷ) = -Σ y_i * log(ŷ_i)
```

Where `y` is the one-hot encoded true label vector and `ŷ` is the vector of predicted probabilities (output of softmax). The log penalizes confident wrong predictions very heavily: if the model assigns 0.01 probability to the correct class, the loss is `-log(0.01) ≈ 4.6`, which is a large penalty.

**Binary Cross-Entropy** is used for two-class problems (or multi-label classification where each output is independent). The formula for one example is:

```
BCE = -(y * log(ŷ) + (1 - y) * log(1 - ŷ))
```

**Categorical Cross-Entropy** expects one-hot encoded labels and is used with softmax outputs for multi-class classification.

**Sparse Categorical Cross-Entropy** is mathematically identical to categorical cross-entropy but expects integer class labels instead of one-hot vectors. It is more memory-efficient when you have many classes (e.g., 1000 ImageNet classes).

### Mean Squared Error: When to Use It

**Mean Squared Error (MSE)** is defined as:

```
MSE = (1/n) * Σ (y_i - ŷ_i)²
```

MSE is the standard loss for **regression** tasks where the output is a continuous value (e.g., predicting house prices, temperature forecasting). It is rarely used for classification because it does not have probabilistic interpretation and can produce poorly calibrated gradient signals. Cross-entropy is almost always preferred for classification.

### Gradient Descent: The Core Optimization Algorithm

Once you have a loss value, you need a strategy to reduce it. **Gradient Descent** is the foundational algorithm. The key insight: the gradient of the loss with respect to each weight `∂L/∂w` tells us the direction of steepest increase. To minimize the loss, we move in the opposite direction.

The weight update rule is:

```
w ← w - η * ∂L/∂w
```

Where `η` (eta) is the **learning rate** — the step size we take in parameter space. If `η` is too large, we overshoot and oscillate or diverge. If too small, training is extremely slow and may get stuck. Typical values range from `0.0001` to `0.1`.

### Stochastic Gradient Descent (SGD)

**Full Batch Gradient Descent** computes the gradient using the entire dataset before updating weights. This gives an accurate gradient but is computationally expensive for large datasets.

**Stochastic Gradient Descent (SGD)** computes the gradient and updates weights after each single training example. This is noisy but fast, and the noise can help escape local minima.

**Mini-Batch Gradient Descent** (the most common in practice) computes the gradient using a small batch (e.g., 32 or 64 examples) before updating. It balances the efficiency of full-batch GD with the noise benefits of SGD, and vectorized operations on GPUs make it highly efficient.

### Momentum: Adding Velocity to Gradient Updates

Plain SGD can oscillate back and forth across narrow valleys in the loss landscape. **Momentum** addresses this by accumulating a velocity vector in the direction of persistent gradients:

```
v ← β * v - η * ∂L/∂w
w ← w + v
```

The momentum term `β` (typically 0.9) makes the update build up speed in consistent directions and damp oscillations. It is analogous to a ball rolling down a hill — it accelerates along the slope and keeps moving even through small bumps.

### Adam Optimizer: Adaptive Moments

**Adam** (Adaptive Moment Estimation) is the most widely used optimizer in deep learning. It combines momentum (first moment) and RMSProp (second moment):

- **First moment (mean of gradients):** `m ← β₁ * m + (1 - β₁) * g`
- **Second moment (uncentered variance):** `v ← β₂ * v + (1 - β₂) * g²`
- **Bias-corrected:** `m̂ = m / (1 - β₁ᵗ)`, `v̂ = v / (1 - β₂ᵗ)`
- **Update:** `w ← w - η * m̂ / (√v̂ + ε)`

Default hyperparameters: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`, `η = 0.001`. Adam adapts the learning rate per parameter: parameters with consistently large gradients get smaller effective learning rates, while infrequently updated parameters get larger updates. This makes Adam excellent for sparse gradients (e.g., NLP embeddings) and deep networks.

### RMSProp: Per-Parameter Learning Rate Adaptation

**RMSProp** (Root Mean Square Propagation) divides the learning rate by an exponential moving average of squared gradients:

```
E[g²]_t ← ρ * E[g²]_{t-1} + (1 - ρ) * g²
w ← w - (η / √(E[g²]_t + ε)) * g
```

This prevents the learning rate from growing too large for parameters with consistently large gradients. RMSProp is effective for recurrent neural networks and was popularized by Geoffrey Hinton.

### Learning Rate Schedules

A fixed learning rate is often suboptimal. **Learning rate schedules** change the learning rate over the course of training:

- **Step Decay:** Reduce the learning rate by a factor (e.g., 0.1) every fixed number of epochs. Simple and effective.
- **Exponential Decay:** `η = η₀ * e^(-k * epoch)`. Smooth continuous reduction.
- **Cosine Annealing:** The learning rate follows a cosine curve from `η_max` to `η_min` and can restart. This allows the optimizer to escape local minima periodically.
- **Warmup + Decay:** Start with a small learning rate, increase to maximum over the first few epochs (warmup), then decay. Popular with Transformers.

### Loss Landscape Visualization

The loss landscape is a high-dimensional surface defined by the loss value at every point in parameter space. Key features:

- **Valleys:** Low-loss regions where the optimizer should settle
- **Saddle points:** Points where gradient is zero but are not minima (flat in some directions, sloping in others). They are problematic for first-order methods.
- **Local minima vs global minimum:** In practice, deep network loss landscapes have many approximately equivalent local minima.
- **Sharp vs flat minima:** Models that converge to flat minima tend to generalize better.

### ASCII Visualization: Gradient Descent

```
Loss
 |
 |   *
 |     *
 |       *
 |         *
 |           * ← Steps converging
 |              *
 |                *
 |                  *___________  ← Minimum
 +----------------------------------------> Weights (w)
```

Each `*` represents a weight update step. Notice how steps are larger at first (steep gradient) and smaller near the minimum (flat gradient).

---

## Key Concepts Table

| Concept | Definition | When to Use |
|---|---|---|
| Cross-Entropy Loss | Measures divergence between predicted and true distributions | Multi-class classification |
| Binary Cross-Entropy | Cross-entropy for 2-class problems | Binary classification, multi-label |
| MSE | Mean of squared prediction errors | Regression |
| Gradient Descent | Iteratively move weights opposite to gradient | All neural network training |
| Learning Rate (η) | Step size for weight updates | Always a critical hyperparameter |
| SGD | Update per single example or mini-batch | General purpose |
| Momentum | Velocity-based SGD smoothing | When SGD oscillates |
| Adam | Adaptive moment estimation optimizer | Default choice for most tasks |
| RMSProp | Per-parameter adaptive learning rate | RNNs, non-stationary objectives |
| Step Decay | Reduce LR every N epochs | Stable training plateaus |
| Cosine Annealing | Cosine LR schedule with optional restarts | State-of-the-art image classification |

---

## Code Reference

```python
import tensorflow as tf

# Compile with different optimizers
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=0.01, decay_steps=1000
)

# Callbacks for learning rate reduction
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=5
)
```

---

## Activities

1. **Learning Rate Sweep:** Train a small CNN on CIFAR-10 with learning rates `[0.1, 0.01, 0.001, 0.0001]`. Run each for 10 epochs with Adam. On a single matplotlib figure, plot the four training loss curves and label each with its learning rate.

2. **Optimizer Comparison:** Using the same CNN architecture and CIFAR-10, train three models: `SGD(lr=0.01)`, `SGD(lr=0.01, momentum=0.9)`, and `Adam(lr=0.001)`. After 10 epochs, plot their validation accuracy curves on the same axes and print the final validation accuracy for each.
## Review Questions

1. Why does using a very high learning rate cause training to diverge?
2. What is the difference between categorical cross-entropy and sparse categorical cross-entropy? When would you use each?
3. Explain the intuition behind momentum. How does it differ from plain SGD?
4. Adam uses two moments of the gradient. What is the purpose of each moment?
5. Why is cross-entropy preferred over MSE for classification tasks?
6. What is a saddle point in the loss landscape? Why can it slow down training?
7. Describe one scenario where you would use a learning rate schedule instead of a fixed learning rate.

---

## Further Reading

- Ruder, S. (2016). *An Overview of Gradient Descent Optimization Algorithms*. arXiv:1609.04747
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, Chapter 8: Optimization
- TensorFlow Keras Optimizers Documentation: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
- Loss Landscape Visualization: https://losslandscape.com/
