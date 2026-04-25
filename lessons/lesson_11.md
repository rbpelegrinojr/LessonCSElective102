# Lesson 11: Loss Functions and Optimizers

## Learning Objectives
- Explain what a loss function measures and why it is essential to the training process
- Derive the intuition behind categorical cross-entropy and when to use it over MSE for classification
- Compare SGD, Momentum, RMSProp, and Adam optimizers and describe the trade-offs of each
- Interpret the effect of learning rate choices on model convergence speed and stability
- Implement learning rate scheduling strategies and explain why they improve training outcomes
- Manually simulate gradient descent to build mechanical intuition for how networks learn

---

## Detailed Explanation

### What Is a Loss Function?

Every time a neural network makes a prediction, it produces some output — a vector of probabilities, a single number, a class label. A **loss function** (also called a cost function or objective function) measures the gap between that prediction and the true answer. Training a neural network is simply the process of minimizing this gap. The loss function is the compass; without it, the network has no signal telling it in which direction to improve.

Think of teaching a student to shoot basketball free throws. After each shot, you measure how far the ball landed from the basket. That distance is your "loss." The student adjusts their technique based on that feedback. Zero loss means a perfect shot every time. The loss function gives precise, quantitative feedback so that adjustments can be made systematically.

Formally, a loss function L takes two inputs — the model's prediction **ŷ** and the true label **y** — and returns a scalar value representing how "wrong" the prediction is:

```
L(y, ŷ) = some measure of error
```

The lower the loss, the better the model is performing. During training, the optimizer uses the gradient of this loss with respect to every weight in the network to update those weights in the direction that reduces the loss.

---

### Categorical Cross-Entropy

For multi-class classification tasks (e.g., identifying digits 0–9, classifying CIFAR-10 images), the most widely used loss function is **categorical cross-entropy**. The formula is:

```
L = -Σ y_i * log(ŷ_i)
```

Where:
- `y_i` is 1 if class `i` is the correct class, 0 otherwise (one-hot encoding)
- `ŷ_i` is the predicted probability for class `i`
- The sum runs over all classes

**Intuition:** If the correct class is class 3, and the model predicts a probability of 0.9 for class 3, then `L = -log(0.9) ≈ 0.105` — a small loss. If the model predicts only 0.1 for class 3, `L = -log(0.1) ≈ 2.303` — a large loss. The logarithm heavily penalizes confident wrong predictions and gently rewards confident correct ones.

Why not use Mean Squared Error (MSE) for classification? MSE treats all errors linearly and doesn't account for the probabilistic nature of softmax outputs. Cross-entropy naturally pairs with softmax because it produces larger gradients when the model is confidently wrong, leading to faster, more stable learning. MSE gradients "vanish" near the boundaries, making learning sluggish.

---

### Binary Cross-Entropy

For binary classification (cat vs. dog, spam vs. not spam), we use **binary cross-entropy**:

```
L = -[y * log(ŷ) + (1 - y) * log(1 - ŷ)]
```

Where `y ∈ {0, 1}` and `ŷ ∈ [0, 1]` is the sigmoid output. This is mathematically equivalent to categorical cross-entropy for two classes.

---

### Mean Squared Error (MSE)

```
L = (1/n) * Σ (y_i - ŷ_i)²
```

MSE is excellent for **regression** problems where outputs are continuous values. It penalizes large errors more strongly (due to squaring) and is differentiable everywhere. However, for classification with softmax, it does not provide the right gradient signal.

---

### Gradient Descent: The Learning Mechanism

Once the loss is computed, the optimizer asks: "How should each weight change to reduce this loss?" The answer comes from **gradient descent**. 

Imagine you are blindfolded on a hilly landscape. You want to reach the lowest valley. Each step, you feel the slope under your feet and take a step in the downhill direction. The steepness of the slope is the gradient; the size of your step is the learning rate.

Mathematically:

```
w_new = w_old - learning_rate * ∂L/∂w
```

The partial derivative `∂L/∂w` tells us how much the loss changes if we nudge weight `w` slightly. By subtracting it (multiplied by the learning rate), we move `w` in the direction that reduces the loss.

**Stochastic Gradient Descent (SGD)** updates weights using the gradient computed on a single training example (or small mini-batch) rather than the entire dataset. This introduces noise that actually helps escape shallow local minima.

---

### Optimizers Compared

| Optimizer | Key Idea | Pros | Cons |
|-----------|----------|------|------|
| SGD | Pure gradient step | Simple, generalizes well | Slow, sensitive to learning rate |
| Momentum | Accumulates velocity in gradient direction | Faster convergence, less oscillation | One extra hyperparameter |
| RMSProp | Divides gradient by running average of squared gradients | Adapts per-parameter, good for RNNs | May not generalize as well |
| Adam | Combines Momentum + RMSProp | Fast convergence, adaptive | Can overfit on small datasets |

**Momentum** adds a "velocity" term that accumulates across steps, similar to a ball rolling down a hill gaining speed. This helps the optimizer blast through flat regions and reduces oscillation in narrow valleys.

**RMSProp** adapts the learning rate per parameter. Parameters with large, frequent gradients get smaller effective learning rates; rare parameters get larger steps. This is especially helpful in recurrent networks where gradients vary wildly.

**Adam (Adaptive Moment Estimation)** maintains both a first moment (mean of gradients, like Momentum) and a second moment (mean of squared gradients, like RMSProp). It corrects for bias in early steps and is considered the default optimizer for most deep learning tasks.

```
Adam update rule (simplified):
m = β1 * m + (1 - β1) * g          # first moment
v = β2 * v + (1 - β2) * g²         # second moment
m̂ = m / (1 - β1^t)                 # bias correction
v̂ = v / (1 - β2^t)                 # bias correction
w = w - lr * m̂ / (sqrt(v̂) + ε)
```

Default values: β1=0.9, β2=0.999, ε=1e-7.

---

### Learning Rate: The Most Critical Hyperparameter

The learning rate controls how large each gradient step is:

```
Too high:  Loss oscillates or diverges (overshooting the minimum)
Too low:   Training is painfully slow; may get stuck in poor minima
Just right: Smooth, steady decrease in loss
```

**Learning Rate Scheduling** strategies:
- **Step Decay**: reduce LR by factor every N epochs
- **Exponential Decay**: LR = LR₀ * exp(-decay_rate * epoch)
- **Cosine Annealing**: LR follows a cosine curve, gently reducing to near-zero
- **Warm Restarts**: periodically reset LR to a high value to escape local minima
- **ReduceLROnPlateau**: reduce LR automatically when validation loss stops improving

A common misconception is that Adam makes learning rate irrelevant. In reality, Adam is far less sensitive to the initial learning rate than SGD, but it still matters — especially for fine-tuning and final training phases.

---

### ASCII Diagram: Loss Landscape

```
Loss
 |
 |    *
 |  *   *
 | *     *
 |*       *    <- saddle point
 |         *
 |          *  *
 |           **
 +-------------------> weights
         ^
    global minimum
```

Gradient descent follows the slope downward at each step. The challenge is that real loss landscapes are high-dimensional and full of saddle points, flat plateaus, and multiple local minima.

---

### Common Misconceptions

1. **"Lower training loss always means a better model"** — No. Lower training loss could mean overfitting. Track both training and validation loss.
2. **"Adam is always the best optimizer"** — Adam trains faster but sometimes generalizes worse than well-tuned SGD with momentum, especially in computer vision.
3. **"Learning rate just controls speed"** — It also controls which minimum the optimizer converges to. A high LR may find a flatter, more generalizable minimum.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Loss Function | Measures the gap between prediction and ground truth | Provides the training signal without which learning is impossible |
| Categorical Cross-Entropy | `-Σ y_i * log(ŷ_i)` for multi-class problems | Standard loss for softmax-based classifiers; provides strong gradient signal |
| Binary Cross-Entropy | `-[y*log(ŷ) + (1-y)*log(1-ŷ)]` | Used for binary classification with sigmoid output |
| Mean Squared Error | Average of squared differences | Used for regression; inappropriate for classification |
| Gradient Descent | Iterative weight update using negative gradient | The fundamental learning algorithm for neural networks |
| Learning Rate | Step size in gradient update | Too high causes divergence; too low causes slow/stuck training |
| SGD | Updates weights using gradients from mini-batches | Simple, effective baseline optimizer; good generalization |
| Momentum | Accumulates gradient history to smooth updates | Faster convergence in consistent gradient directions |
| RMSProp | Adapts learning rate using running average of squared gradients | Handles non-stationary gradients well |
| Adam | Combines Momentum and RMSProp with bias correction | Fast, robust, widely used default optimizer |
| Learning Rate Schedule | Strategy for changing LR over training | Improves convergence and final accuracy |
| Saddle Point | Point where gradient is zero but is not a minimum | Can trap gradient descent in flat, non-optimal regions |

---

## Code Reference

See the full runnable demo in **code/lesson_11.py**

---

## Activities

1. **Loss Comparison Experiment**: Train two identical models — one with MSE loss and one with categorical cross-entropy — on MNIST. Plot training accuracy and loss for both. Write a paragraph explaining which performs better and why.

2. **Optimizer Race**: Train the same CNN architecture on CIFAR-10 using SGD (lr=0.01), Adam (lr=0.001), and RMSProp (lr=0.001). Plot validation accuracy over 15 epochs for all three on the same graph. Which converges fastest? Which achieves the highest final accuracy?

3. **Learning Rate Explorer**: Using MNIST, train with learning rates [0.1, 0.01, 0.001, 0.0001]. Plot the training loss curves on one figure. Describe what happens at each learning rate and identify the "Goldilocks" zone.

4. **Manual Gradient Descent**: Implement gradient descent from scratch (no Keras) on a simple quadratic function `f(x) = x² + 2x + 1`. Plot the path of the point `x` over 50 iterations for learning rates 0.1, 0.5, and 1.5. Which diverges?

5. **Learning Rate Scheduling**: Train a CNN on CIFAR-10 with a fixed learning rate vs. one with `ReduceLROnPlateau` (patience=3, factor=0.5). Compare final accuracy and training stability. When does the learning rate actually reduce?

---

## Review Questions

1. Why is categorical cross-entropy preferred over MSE for classification tasks? Explain using the gradient signal argument.
2. Describe the Adam optimizer's update rule in plain English. What problem does the bias correction step solve?
3. What happens to training if the learning rate is set too high? Sketch a hypothetical loss curve to illustrate.
4. Explain the "hill-climbing analogy" for gradient descent. What feature of the landscape does the gradient represent, and what does the learning rate control?
5. Compare SGD with momentum and vanilla SGD. Under what circumstances does momentum help the most, and when might it hurt?

---

## Further Reading

- "An Overview of Gradient Descent Optimization Algorithms" — Sebastian Ruder (ruder.io)
- "Adam: A Method for Stochastic Optimization" — Kingma & Ba, 2014 (arXiv:1412.6980)
- Deep Learning Book, Chapter 8: Optimization for Training Deep Models — Goodfellow, Bengio, Courville
- "Cyclical Learning Rates for Training Neural Networks" — Leslie N. Smith, 2017
- Keras documentation: `keras.optimizers` and `keras.losses` modules
