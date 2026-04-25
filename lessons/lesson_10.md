# Lesson 10: Training a CNN

## Learning Objectives

By the end of this lesson, you will be able to:

- Describe every step of the CNN training loop (forward pass, loss, backward pass, weight update)
- Distinguish between epochs, batches, and iterations
- Explain the role of the learning rate in training
- Use Keras `model.fit()` with all key parameters
- Interpret training and validation curves to diagnose model health
- Implement EarlyStopping and ModelCheckpoint callbacks

---

## Detailed Explanation

### The Training Loop: How a CNN Learns

Training a neural network is an iterative optimization process. The goal is to find the weights W that minimize a loss function L(W), which measures how wrong the model's predictions are. This optimization is performed by repeatedly exposing the model to training data and updating weights in the direction that reduces loss.

The full training loop for one iteration:

```
┌─────────────────────────────────────────────────────────┐
│                   ONE TRAINING ITERATION                │
│                                                         │
│  1. FORWARD PASS                                        │
│     Input batch (x, y) → Conv layers → Dense → ŷ       │
│                                                         │
│  2. LOSS COMPUTATION                                    │
│     L = CrossEntropy(ŷ, y)                              │
│     (compare predictions to ground truth)               │
│                                                         │
│  3. BACKWARD PASS (Backpropagation)                     │
│     ∂L/∂W = chain rule through all layers               │
│     (compute gradient of loss w.r.t. every weight)      │
│                                                         │
│  4. WEIGHT UPDATE                                       │
│     W ← W - learning_rate × ∂L/∂W                      │
│     (move weights in direction of decreasing loss)      │
└─────────────────────────────────────────────────────────┘
```

This four-step cycle repeats for every batch in the training set, and the full pass over the training set is called an **epoch**.

---

### Forward Pass

During the forward pass, a batch of input images flows through the network layer by layer. Each layer applies its learned transformation (convolution, activation, pooling, dense multiplication) until the final layer produces predictions ŷ — a probability distribution over classes.

At initialization, weights are small random values (Xavier/He initialization). The first forward pass produces essentially random predictions. The entire training process is about systematically adjusting weights to make better predictions.

---

### Loss Computation

The loss function quantifies the error between predictions ŷ and ground truth labels y. For classification:

**Cross-entropy loss:**
```
L = -1/N × Σ y_i × log(ŷ_i)
```

For multi-class problems, this collapses to: L = -log(ŷ[true_class])

The loss is high when the model assigns low probability to the correct class, and low when it assigns high probability. A loss of 0 would mean perfect predictions; a loss of log(N) (where N is the number of classes) corresponds to uniform, completely random predictions.

---

### Backward Pass: Backpropagation

Backpropagation computes the gradient of the loss with respect to every weight in the network using the chain rule of calculus. Starting from the output layer and working backward:

```
∂L/∂W_L = ∂L/∂ŷ × ∂ŷ/∂W_L                    (output layer)
∂L/∂W_{L-1} = ∂L/∂ŷ × ∂ŷ/∂h_L × ∂h_L/∂W_{L-1} (second-to-last layer)
... and so on back to the first layer
```

This gradient tells us: if we increase weight W by a tiny amount ε, the loss increases by approximately ε × (∂L/∂W). So to decrease the loss, we should decrease weights with positive gradients and increase weights with negative gradients.

TensorFlow's automatic differentiation (`tf.GradientTape`) handles this computation automatically — you never need to derive gradients by hand.

---

### Weight Update: The Optimizer

The optimizer uses the computed gradients to update weights. The simplest update rule is Stochastic Gradient Descent (SGD):

```
W ← W - η × ∂L/∂W
```

where η (eta) is the **learning rate** — the most important hyperparameter in all of deep learning.

**Adam optimizer** (Adaptive Moment Estimation) is more sophisticated. It maintains:
- A running average of past gradients (momentum) — smooths out noisy updates
- A running average of squared gradients (RMSProp) — adapts step size per parameter

This makes Adam much more robust to learning rate choice and converges faster than vanilla SGD in most practical settings.

---

### Epochs, Batches, and Iterations

These three terms are frequently confused:

| Term      | Definition                                              | Example                              |
|-----------|---------------------------------------------------------|--------------------------------------|
| Sample    | A single training example                              | 1 image                              |
| Batch     | A group of samples processed together                  | 32 images                            |
| Iteration | One forward + backward pass over one batch             | 1 gradient update                    |
| Epoch     | One complete pass over the entire training set          | All N training images seen once      |

For a dataset of 50,000 images with batch_size=32:
```
Steps per epoch = ceil(50,000 / 32) = 1,563 iterations per epoch
```

After 10 epochs, you've done 15,630 weight updates total.

**Why not use the full dataset in one batch?** Larger batches require more memory and, surprisingly, often generalize worse. Mini-batch gradient descent (batch sizes of 16–256) introduces beneficial noise that helps escape sharp local minima.

---

### Learning Rate: The Most Important Hyperparameter

The learning rate η controls how large each weight update step is.

**Too high:** The optimizer overshoots the minimum. Loss oscillates or diverges (increases). The model never converges.

**Too low:** Training proceeds correctly but extremely slowly. You may need 10× more epochs to reach the same accuracy.

**Just right:** Loss decreases steadily, validation accuracy improves, training completes in reasonable time.

**Typical starting values:**
- Adam optimizer: 0.001 (default)
- SGD with momentum: 0.01–0.1
- Fine-tuning pretrained models: 0.0001 (10× smaller)

**Learning rate schedules** reduce the learning rate over time, allowing large steps early (fast progress) and small steps later (fine-tuning). Common schedules: step decay, cosine annealing, exponential decay.

---

### Keras model.fit() — All Parameters Explained

```python
history = model.fit(
    x=train_data,           # Training data (numpy array or tf.data.Dataset)
    y=train_labels,         # Labels (None if using tf.data.Dataset)
    batch_size=32,          # Samples per gradient update
    epochs=20,              # Number of complete passes over training data
    validation_data=(X_val, y_val),  # Validation data (monitored but not trained on)
    validation_split=0.1,   # Alternatively, use 10% of training data for validation
    shuffle=True,           # Shuffle training data each epoch
    callbacks=[...],        # List of Callback objects (EarlyStopping, ModelCheckpoint)
    verbose=1               # 0=silent, 1=progress bar, 2=one line per epoch
)
```

The `history` object returned by `model.fit()` contains the loss and metric values for each epoch, which you can plot to visualize training progress.

---

### Training Curves: Healthy vs Unhealthy

The most informative diagnostic in deep learning is the training and validation curve over epochs.

**Healthy training:**
```
Loss                              Accuracy
1.5 |\.                           0.5 |     ...------
1.0 | `\.                         0.7 |   ..
0.5 |   `\....---                 0.9 |./
    +──────────────── epochs          +────────── epochs
    — train   - - val                 — train   - - val
```
Both training and validation metrics improve together, with a small gap (validation slightly worse than training). This is the ideal case.

**Overfitting:**
```
Loss
1.5 |\.        _/---  ← val loss rising
1.0 | `\------
0.5 |           ← train loss still decreasing
    +────────── epochs
```
Training loss keeps decreasing but validation loss starts increasing. The model is memorizing training data instead of learning generalizable patterns.

**Underfitting:**
```
Loss
2.0 |\.
1.8 | `\..........  ← both plateau high
    +────────── epochs
```
Both losses plateau at a high value. The model lacks capacity or is not training long enough.

**Unstable training:**
```
Loss
2.0 |\/\/\/\/\/\/  ← noisy oscillations
    +────────── epochs
```
Usually indicates a learning rate that is too high.

---

### Common Training Problems

- **Loss not decreasing:** Check learning rate (try 10×smaller), check data normalization, check loss function matches output activation
- **Accuracy stuck at 1/N (random):** The model may have all-zero or all-NaN weights; check for gradient explosion; verify input shapes
- **Validation much worse than training:** Overfitting — reduce model complexity, add dropout/regularization, get more data
- **NaN loss:** Learning rate too high causing gradient explosion; also check for zero or negative values passed to log()

---

### GPU vs CPU Considerations

TensorFlow automatically uses GPU when available. Key considerations:
- GPU batch sizes: use powers of 2 (32, 64, 128, 256) to maximize GPU utilization
- Mixed precision training (`tf.keras.mixed_precision`) uses float16 on GPU for 2–3× speedup
- `model.fit()` on CPU can be 10–100× slower than GPU for large CNNs

---

### Callbacks: EarlyStopping and ModelCheckpoint

**EarlyStopping:** Monitors a metric (usually validation loss). Stops training if it doesn't improve for `patience` epochs. Optionally restores the best weights.

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

**ModelCheckpoint:** Saves the model (or just weights) whenever a monitored metric improves.

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.keras',
    monitor='val_accuracy',
    save_best_only=True
)
```

Using both together is best practice: ModelCheckpoint saves the best model, EarlyStopping prevents wasting time after performance peaks.

---

## Key Concepts Table

| Term                   | Definition                                                                  |
|------------------------|-----------------------------------------------------------------------------|
| Forward pass           | Input flowing through the network to produce predictions                    |
| Loss function          | Mathematical measure of prediction error                                    |
| Backpropagation        | Algorithm to compute gradients of loss w.r.t. all weights                  |
| Optimizer              | Algorithm that uses gradients to update weights (SGD, Adam)                 |
| Learning rate          | Step size scalar for weight updates; most critical hyperparameter            |
| Epoch                  | One complete pass through the entire training dataset                       |
| Batch                  | Subset of data processed together in one gradient update                    |
| Overfitting            | Model learns training data patterns but fails to generalize                 |
| EarlyStopping          | Callback that halts training when validation performance stops improving    |

---

## Code Reference

```python
import tensorflow as tf

# Compile
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                     restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint('best.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                         patience=3, verbose=1)
]

# Train
history = model.fit(X_train, y_train,
                    batch_size=64,
                    epochs=50,
                    validation_data=(X_val, y_val),
                    callbacks=callbacks,
                    verbose=1)

# Plot training curves
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
```

---

## Activities

### Activity 10.1 — Manual Training Loop
Implement a training loop from scratch using only NumPy on a simple binary classification problem. Print the loss at each epoch and observe it decreasing.

### Activity 10.2 — Keras Training
Train the CNN from Lesson 8 on MNIST using `model.fit()`. Use all parameters discussed in this lesson. What final validation accuracy do you achieve?

### Activity 10.3 — Curve Analysis
Plot training and validation accuracy/loss curves from Activity 10.2. Identify whether the model is overfitting, underfitting, or training healthily.

### Activity 10.4 — Learning Rate Experiment
Run Section 4 of `code/lesson_10.py`. Train the same network with learning rates [0.1, 0.01, 0.001, 0.0001]. Plot all four loss curves. Which learning rate converges fastest and most stably?

### Activity 10.5 — Callbacks
Add EarlyStopping with patience=3 to your training run from Activity 10.2. Does it stop before your specified number of epochs? What was the best validation accuracy?

---

## Review Questions

1. Describe the four steps of one training iteration in a neural network.
2. What is the difference between an epoch and an iteration?
3. Why is the learning rate considered the most important hyperparameter?
4. What does a rising validation loss combined with a falling training loss indicate?
5. In Keras's `model.fit()`, what is the difference between `validation_split` and `validation_data`?
6. What does EarlyStopping's `patience` parameter control?
7. Why is mini-batch gradient descent typically preferred over full-batch gradient descent?
8. What would you check first if your model's loss becomes NaN after a few training steps?

---

## Further Reading

- [Backpropagation Explained — Andrej Karpathy](https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b)
- [An Overview of Gradient Descent Optimization Algorithms — Sebastian Ruder](https://ruder.io/optimizing-gradient-descent/)
- [Adam: A Method for Stochastic Optimization — Kingma & Ba, 2014](https://arxiv.org/abs/1412.6980)
- [A Disciplined Approach to Neural Network Hyper-Parameters — Leslie Smith](https://arxiv.org/abs/1803.09820)
- [Keras Training & Evaluation Guide](https://keras.io/guides/training_with_built_in_methods/)
