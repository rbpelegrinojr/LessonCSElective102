# Lesson 10: Training a CNN

## Learning Objectives
- Trace the complete training loop: forward pass, loss calculation, backpropagation, and weight update
- Distinguish between batch gradient descent, stochastic gradient descent, and mini-batch gradient descent
- Explain the role of learning rate and describe symptoms of learning rates that are too high or too low
- Implement and configure Keras callbacks: ModelCheckpoint, EarlyStopping, and ReduceLROnPlateau
- Interpret learning curves to diagnose underfitting, overfitting, and convergence
- Describe the practical differences between training on CPU and GPU

---

## Detailed Explanation

### The Training Loop in Detail

Training a neural network is an iterative optimization process. At its heart, every training step follows the same four-stage cycle:

**Stage 1 — Forward Pass:**
The input batch flows through the network layer by layer. Each layer applies its current weights and activation function. The final layer produces predictions (e.g., probability distributions over 10 classes). At this point, the model's weights haven't changed — we're just computing what the *current* model predicts.

```
Input Batch → [Conv → ReLU → Pool] × N → Flatten → Dense → Softmax → Predictions
```

**Stage 2 — Loss Calculation:**
The predictions are compared to the true labels using a loss function. For multi-class classification, categorical cross-entropy is used:

```
Loss = -sum(y_true * log(y_pred))
```

Where y_true is the one-hot true label and y_pred is the predicted probability distribution. A perfect prediction (probability 1.0 for the correct class) gives loss = -log(1.0) = 0. A wrong prediction (probability 0.01 for the correct class) gives loss = -log(0.01) ≈ 4.6.

**Stage 3 — Backpropagation:**
The loss is differentiated with respect to each weight in the network using the chain rule of calculus. This produces a **gradient** for each parameter — a vector pointing in the direction that *increases* the loss. We want to *decrease* the loss, so we move in the opposite direction.

The chain rule means gradients flow backward through the network layer by layer:
```
∂Loss/∂w_early = ∂Loss/∂y_final × ∂y_final/∂y_deep × ... × ∂y_early/∂w_early
```

This is why activation function choice matters: if any gradient in this chain is near zero (vanishing gradient), the early weights receive no useful signal.

**Stage 4 — Weight Update:**
Each weight is updated by subtracting a fraction of its gradient:

```
w_new = w_old - learning_rate × gradient
```

This fraction is the **learning rate** — one of the most critical hyperparameters in the entire training process.

---

### Epochs, Iterations, and Steps

These three terms describe the same training process at different granularities, and confusing them is a common source of errors:

**Epoch:** One complete pass through the entire training dataset. If you have 50,000 training images and train for 20 epochs, the model sees each image 20 times.

**Iteration (step):** One forward+backward pass using one batch. With batch size 32 and 50,000 images: `50,000 / 32 ≈ 1,563 iterations per epoch`.

**Steps per epoch:** Keras terminology for iterations per epoch. Often calculated automatically, but can be set manually:
```python
steps_per_epoch = len(x_train) // batch_size
```

**Relationship summary:**
```
1 Epoch = N iterations = N batches processed
N = ceil(dataset_size / batch_size)
```

---

### Gradient Descent Variants

There are three main variants of gradient descent, differing in how much data each uses per weight update:

**Batch Gradient Descent (BGD):**
Uses the *entire dataset* to compute gradients before updating weights. Produces accurate gradient estimates but requires loading all data into memory and is slow for large datasets. Rarely used in deep learning.

**Stochastic Gradient Descent (SGD, pure):**
Uses a *single sample* per weight update. Very fast but extremely noisy — the gradient estimate from one sample is a poor approximation of the true gradient. The noise can actually help escape local minima but makes convergence erratic.

**Mini-Batch Gradient Descent:**
Uses a small batch (typically 32–256 samples) per update. This is the standard approach in deep learning. It balances:
- Efficient GPU utilization (batches fill GPU memory efficiently)
- Reasonable gradient estimate quality (better than single-sample)
- Implicit regularization from gradient noise (better than full-batch)

| Method | Speed | Memory | Gradient Quality | Generalization |
|--------|-------|--------|-----------------|----------------|
| Full Batch | Slow | High | Exact | Poor (sharp minima) |
| Mini-Batch (32-256) | Fast | Medium | Good estimate | Best |
| Single Sample (SGD) | Very Fast | Minimal | Noisy | Mediocre |

---

### The Learning Rate — The Most Important Hyperparameter

The learning rate `α` controls how large each weight update step is. Getting it right is critical:

**Learning rate too high:**
- Large gradient steps overshoot the minimum
- Loss oscillates wildly or diverges (increases instead of decreasing)
- Symptom: training loss bounces up and down; may exceed initial value

**Learning rate too low:**
- Tiny gradient steps barely move the weights
- Training converges extremely slowly
- Symptom: loss decreases very slowly; still high after many epochs

**Good learning rate:**
- Loss decreases smoothly and steadily
- Converges to a good solution in a reasonable number of epochs

```
Ideal Learning Curve:
Loss
4.0 |*
3.0 |  *
2.0 |    **
1.0 |      ****
0.5 |          *****
0.2 |               *****
    +-------------------------> Epoch
```

**Learning rate schedules:** The optimal learning rate changes during training. Start with a moderate rate (0.001 is a common default for Adam) and reduce it over time as you approach the minimum:
- `ReduceLROnPlateau`: reduce LR when validation loss stops improving
- `CosineAnnealing`: smoothly decay LR following a cosine curve
- `WarmupSchedule`: start very low, ramp up, then decay (used in transformers)

---

### Keras Callbacks

Callbacks are functions that are called at specific points during training. They allow you to monitor and control the training process:

**ModelCheckpoint:**
Saves the model to disk during training. Crucially, you can save only the *best* model (by validation loss), not the final model — because the final model may have overfit:

```python
checkpoint = ModelCheckpoint(
    'best_model.keras',
    monitor='val_loss',
    save_best_only=True,  # only save when val_loss improves
    verbose=1
)
```

**EarlyStopping:**
Stops training when validation performance stops improving. This prevents wasting time on epochs that only increase overfitting:

```python
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,      # wait 5 epochs for improvement before stopping
    restore_best_weights=True  # revert to best weights, not final weights
)
```

The `patience` parameter is critical: too low and you stop prematurely before convergence; too high and you waste time overfitting.

**ReduceLROnPlateau:**
Reduces the learning rate when validation loss plateaus. This often triggers a second phase of improvement after the initial rapid learning:

```python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # multiply LR by 0.5 when triggered
    patience=3,       # wait 3 epochs of no improvement
    min_lr=1e-6       # don't reduce below this
)
```

A common training trajectory with ReduceLROnPlateau:
```
Phase 1: LR=0.001 → rapid initial improvement
Phase 2: LR=0.0005 → second phase of improvement (after plateau)
Phase 3: LR=0.00025 → fine-tuning
Phase 4: EarlyStopping triggers → done
```

---

### Understanding Learning Curves

The learning curves — plots of training and validation loss (or accuracy) over epochs — are your primary diagnostic tool:

**Healthy training (good generalization):**
```
Loss
│ ╲  training loss
│  ╲
│   ╲── validation loss (slightly above training)
│    ╲___________
│               converged
└─────────────────────── Epoch
```

**Overfitting:**
```
Loss
│ ╲  training loss (keeps decreasing)
│  ╲___________
│
│    validation loss (starts rising)
│         /
│        /
└─────────────────────── Epoch
```

**Underfitting:**
```
Loss
│
│ ─────────── training loss (not decreasing enough)
│ ─────────── validation loss (similar to training)
│
└─────────────────────── Epoch
```

**What "convergence" means:**
A model has converged when its loss has stabilized and is no longer meaningfully improving. This doesn't mean loss is zero — it means further training is unlikely to significantly improve performance. Signs of convergence:
- Loss curve becomes approximately flat (horizontal)
- Gradient magnitudes become very small
- Validation loss matches training loss closely (no growing gap)

---

### What Happens During Each Epoch

```
Epoch Start
     ↓
Shuffle training data
     ↓
For each mini-batch:
  ├── Forward pass (predict)
  ├── Calculate loss
  ├── Backward pass (compute gradients)
  └── Update weights
     ↓
Evaluate on validation set (no weight updates)
     ↓
Execute callbacks (EarlyStopping check, LR reduction, model save)
     ↓
Print metrics (loss, accuracy for train and val)
     ↓
Epoch End → Next Epoch (or stop)
```

---

### Training on CPU vs. GPU

CNNs perform millions of floating-point multiplications per forward pass. GPUs are designed for exactly this — thousands of small cores that perform arithmetic operations in parallel — while CPUs have a few powerful cores optimized for sequential tasks.

**Speedup:** A modern GPU (e.g., NVIDIA RTX 3090) can be 20–100× faster than a CPU for training CNNs. CIFAR-10 training that takes 2 minutes on a GPU might take over an hour on CPU.

**Memory consideration:** The entire batch must fit in GPU VRAM. If batch size is too large, you'll get an OOM (out-of-memory) error. Solutions: reduce batch size, use mixed precision (float16), or use gradient checkpointing.

**In Keras:** No code changes are needed — TensorFlow automatically uses GPU if one is available. You can verify with:
```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Forward Pass | Computing predictions by flowing input through the network | Produces the output needed to calculate loss |
| Backpropagation | Computing gradients of loss with respect to each weight using chain rule | Tells us how to change each weight to reduce loss |
| Learning Rate | Scalar that scales the gradient during weight update | Too high = divergence; too low = slow convergence |
| Mini-Batch | Small subset of training data used for each weight update | Balances computational efficiency and gradient quality |
| Epoch | One complete pass through all training data | Unit for measuring training progress |
| Convergence | State where loss has stabilized and training improvements have plateaued | Indicates training is complete or near-complete |
| ModelCheckpoint | Callback that saves model weights during training | Preserves best model before overfitting occurs |
| EarlyStopping | Callback that halts training when validation loss stops improving | Prevents overfitting and wasted computation |
| ReduceLROnPlateau | Callback that reduces learning rate when progress stalls | Often triggers second improvement phase |
| Learning Curve | Plot of training/validation loss and accuracy over epochs | Primary diagnostic tool for training health |
| Overfitting | Training loss decreasing while validation loss increases | Model is memorizing training data, not generalizing |

---

## Code Reference

See [`code/lesson_10.py`](../code/lesson_10.py) for fully runnable CNN training on CIFAR-10 with all callbacks, learning curve visualization, batch size comparison, and model save/reload.

---

## Activities

1. **Learning Rate Exploration:** Train the same CNN on CIFAR-10 three times with learning rates: 0.1, 0.001, and 0.00001. Plot all three training loss curves on the same graph. Identify which is too high, too low, and approximately right. What visual characteristics distinguish each case?

2. **Callback Investigation:** Train a CNN with `EarlyStopping(patience=3)` and `ModelCheckpoint(save_best_only=True)`. After training stops, load the saved best model and evaluate it on the test set. Compare this to evaluating the final (last epoch) model. Is there a difference?

3. **Batch Size Comparison:** Train the same architecture four times with batch sizes: 16, 32, 128, 512. Record: training time per epoch, final training accuracy, final validation accuracy. Plot learning curves. What tradeoffs do you observe? (Hint: smaller batches are noisier but can generalize better.)

4. **Loss Curve Diagnosis:** Below are three described learning curve patterns. For each, diagnose the problem and suggest a specific fix: (a) Both train and val loss are high and decreasing very slowly after 50 epochs. (b) Train loss is near 0 but val loss is 3× higher and rising. (c) Both train and val loss oscillate wildly (spike up and down each epoch).

5. **Callbacks from Scratch:** Without using Keras callbacks, implement early stopping manually by writing a training loop using `model.train_on_batch()`. Track validation loss after each epoch and stop training if it hasn't improved for 5 consecutive epochs. Compare results to Keras's built-in EarlyStopping.

---

## Review Questions

1. Describe the four stages of one training iteration (forward pass, loss, backward pass, weight update) in your own words. What is the role of the learning rate in stage 4?

2. What is the difference between an epoch, an iteration, and a step? If you have 45,000 training samples and batch size 32, how many iterations are in one epoch?

3. Explain the difference between batch gradient descent, pure SGD, and mini-batch gradient descent. Why is mini-batch the standard choice for deep learning?

4. What does the `patience` parameter in `EarlyStopping` control? What are the risks of setting it too low? Too high?

5. Looking at a learning curve where training accuracy reaches 98% but validation accuracy plateaus at 72%, what is happening? List three techniques you could try to improve the situation.

---

## Further Reading

- **"An overview of gradient descent optimization algorithms"** (Ruder, 2016) — Comprehensive survey of SGD variants including Adam, RMSprop, and momentum
- **"Cyclical Learning Rates for Training Neural Networks"** (Smith, 2017) — Introduces the learning rate range test and cyclical LR schedules
- **Keras Callbacks documentation** — https://keras.io/api/callbacks/ — Complete reference for all built-in callbacks
- **"Don't Decay the Learning Rate, Increase the Batch Size"** (Smith et al., 2018) — Interesting perspective on the LR vs. batch size tradeoff
- **"A Recipe for Training Neural Networks"** — Blog post by Andrej Karpathy covering practical training tips and common failure modes
