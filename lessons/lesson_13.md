# Lesson 13: Overfitting & Regularization Techniques

## Learning Objectives

By the end of this lesson, you will be able to:

- Define overfitting and underfitting and identify them from training curves
- Explain the bias-variance tradeoff and its implications for model design
- Apply L1 and L2 regularization to Keras layers
- Implement and tune Dropout to reduce overfitting
- Use early stopping to automatically halt training at the right moment

---

## Detailed Explanation

### What Is Overfitting?

**Overfitting** occurs when a model learns the training data too well — including its noise, outliers, and irrelevant patterns — to the point where it fails to generalize to new, unseen data. The model has essentially **memorized** the training set rather than learning the underlying patterns.

Signs of overfitting:
- Training accuracy is high (e.g., 99%), but validation accuracy is significantly lower (e.g., 75%)
- Training loss keeps decreasing while validation loss begins to increase or plateaus
- The model performs well on training data but poorly in real-world deployment

Consider a high school student who memorizes every question from past exams without understanding the material. They score perfectly on those exact questions, but fail any question phrased differently. That's overfitting.

### What Is Underfitting?

**Underfitting** occurs when a model is too simple to capture the underlying patterns in the data. Both training and validation performance are poor. This happens when:

- The model has too few parameters (too shallow/narrow)
- Training has not run long enough
- The learning rate is too small or too large
- The features provided are insufficient or uninformative

Continuing the student analogy: underfitting is like a student who barely studied and cannot answer even the simplest questions.

### The Bias-Variance Tradeoff

This is one of the most fundamental concepts in machine learning:

- **Bias:** The error introduced by simplifying assumptions. A high-bias model is underfit — it consistently misses the target pattern.
- **Variance:** The sensitivity of the model to fluctuations in training data. A high-variance model is overfit — it changes drastically when trained on different subsets of data.

The total expected error decomposes as: `Error = Bias² + Variance + Irreducible Noise`

**Analogy:** Imagine throwing darts at a target:
- High bias, low variance: All darts cluster together but far from the bullseye (consistently wrong)
- Low bias, high variance: Darts spread all over the board (sometimes right, usually scattered)
- Low bias, low variance: Darts cluster at the bullseye ✅ (what we want)

Increasing model complexity reduces bias but increases variance. The art of machine learning is finding the sweet spot between these two extremes.

### The Learning Curve: Training vs Validation Accuracy

The **learning curve** plots training and validation accuracy (or loss) as a function of training epochs or dataset size. It is the primary diagnostic tool for overfitting:

```
Accuracy
  |
  |  train ────────────────
  |                        \
  |  val  ─────────\        \  ← validation drops: overfitting
  |                 \_______
  +-----------------------------------> Epochs
```

In a healthy training run, training accuracy slightly exceeds validation accuracy, but both trend upward together. When they diverge sharply, overfitting is occurring.

### L1 Regularization (Lasso)

**L1 regularization** adds the sum of absolute values of weights to the loss:

```
L_total = L_original + λ * Σ |w_i|
```

The effect: L1 regularization drives many weights to exactly zero, producing **sparse models**. This is useful for feature selection — only the most important features survive. The parameter `λ` (lambda) controls the regularization strength: larger `λ` = stronger penalty = sparser weights.

In Keras:
```python
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l1(0.01))
```

### L2 Regularization (Ridge / Weight Decay)

**L2 regularization** (also called **weight decay** in the context of neural networks) adds the sum of squared weights to the loss:

```
L_total = L_original + λ * Σ w_i²
```

Unlike L1, L2 pushes weights toward zero but rarely to exactly zero. It penalizes large weights more heavily than small ones (because of the squaring), effectively preventing any single weight from becoming dominant. L2 is the most common regularization technique for deep learning and is equivalent to placing a Gaussian prior on the weights.

In Keras:
```python
tf.keras.layers.Dense(64, kernel_regularizer=tf.keras.regularizers.l2(0.01))
```

L1+L2 combined is called **Elastic Net**: `λ₁ * Σ|w| + λ₂ * Σw²`

### Dropout: Randomly Zero Out Neurons During Training

**Dropout** is one of the most effective and widely used regularization techniques for neural networks. During each forward pass in training, each neuron is independently set to zero with probability `p` (the dropout rate). Typically `p = 0.2` to `p = 0.5` for hidden layers.

```
Layer output → [a₁, a₂, a₃, a₄, a₅]
                  ↓  (dropout p=0.4)
              [a₁,  0, a₃,  0, a₅]   ← 2 neurons dropped randomly
```

**Why dropout works:** Several complementary interpretations:
1. **Ensemble interpretation:** Each forward pass trains a different sub-network (with a random subset of neurons). The full model is approximately an ensemble of 2^n sub-networks, which tend to generalize better.
2. **Co-adaptation prevention:** Neurons cannot rely on specific other neurons always being present, so each must learn more robust, independent features.
3. **Noise injection:** Adding random noise to activations acts as data augmentation in feature space.

During **inference**, all neurons are active. To maintain the same expected activation magnitude, weights are scaled by `(1 - p)`. Modern implementations use **inverted dropout** during training: divide active neurons by `(1 - p)` during training so no rescaling is needed at inference time.

### Early Stopping: Halt Training When Validation Loss Stops Improving

**Early stopping** monitors a validation metric (typically validation loss) and stops training when it no longer improves for a specified number of epochs (called `patience`):

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)
```

With `restore_best_weights=True`, the model weights are rolled back to the epoch with the best validation performance, even if training continued for several more epochs afterward. Early stopping effectively defines the model's capacity in the time dimension — it prevents the model from training long enough to overfit.

### Data Augmentation as Implicit Regularization

By artificially generating new training examples from existing ones (flipping, rotating, cropping, etc.), data augmentation increases the effective size and diversity of the training set. The model sees slightly different versions of each image in each epoch, making it harder to memorize any single example. This is covered in depth in Lesson 14.

### Model Capacity and Complexity

**Model capacity** refers to the richness of functions a model can represent. A model with more layers, more filters, or more neurons per layer has higher capacity:

- Too low capacity → underfitting (cannot learn the pattern)
- Too high capacity → overfitting (learns noise and specific training examples)

As a rule of thumb: **start small and increase capacity only if validation loss plateaus without overfitting**. Adding regularization allows you to use higher-capacity models while controlling generalization.

### ASCII Diagram: Overfit vs Underfit vs Good Fit

```
       Underfit          Good Fit          Overfit
  
       * *               * *               *
  *                   *     *           *
       *           *           *            *
   *      *     *                 *   *
               (smooth curve)    (wiggly, hugs every point)
  
  High Bias          Balanced          High Variance
  Low Variance                         Low Bias
```

---

## Key Concepts Table

| Technique | How It Works | Hyperparameter |
|---|---|---|
| L1 Regularization | Adds `λ * Σ\|w\|` to loss; sparse weights | `λ` (l1 factor) |
| L2 Regularization | Adds `λ * Σw²` to loss; shrinks all weights | `λ` (l2 factor) |
| Dropout | Randomly zeros neurons during training | `p` (dropout rate) |
| Early Stopping | Halts training when validation stops improving | `patience` (epochs) |
| Weight Decay | L2 penalty applied in optimizer (AdamW) | `weight_decay` |
| Data Augmentation | Artificial data diversity during training | Augmentation params |

---

## Code Reference

```python
import tensorflow as tf

# L1 and L2 regularization
model = tf.keras.Sequential([
    tf.keras.layers.Dense(256, activation='relu',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Early stopping callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history = model.fit(X_train, y_train,
    validation_data=(X_val, y_val),
    callbacks=[early_stop], epochs=100)
```

---

## Activities

1. **Overfitting Playground:** Use a very deep MLP on a tiny subset (500 examples) of CIFAR-10. Watch the training/validation gap widen. Then add dropout layers and compare the curves.

2. **L2 Lambda Search:** Train the same model with `λ ∈ {0, 0.0001, 0.001, 0.01, 0.1}`. Plot validation accuracy vs. lambda. Find the sweet spot.

3. **Early Stopping Investigation:** Train a model for 200 epochs. Log when early stopping would have triggered (patience=10). Compare the final model to the best-saved model.

4. **Bias-Variance Decomposition:** Train 10 identical models on different random subsets of data. Measure prediction variance across models and relate to the overfitting/underfitting behavior.

---

## Review Questions

1. A model achieves 98% training accuracy and 60% validation accuracy. What is happening? List three techniques to address this.
2. Explain the bias-variance tradeoff in your own words using an analogy.
3. Why does L1 regularization produce sparse weights while L2 does not?
4. During inference with dropout, why don't we drop neurons? How is the output magnitude kept consistent?
5. What is the `patience` parameter in early stopping? What happens if it is set too low?
6. A model achieves only 55% training accuracy after 50 epochs. Is this overfitting or underfitting? What would you try first?
7. Why is data augmentation considered a form of regularization?

---

## Further Reading

- Srivastava, N., et al. (2014). *Dropout: A Simple Way to Prevent Neural Networks from Overfitting*. JMLR.
- Ng, A. (2004). *Feature selection, L1 vs L2 regularization, and rotational invariance*. ICML.
- Goodfellow, I., Bengio, Y., & Courville, A. *Deep Learning*, Chapter 7: Regularization
- Prechelt, L. (1998). *Early Stopping - But When?*. Neural Networks: Tricks of the Trade.
- Loshchilov, I., & Hutter, F. (2019). *Decoupled Weight Decay Regularization* (AdamW). ICLR.
