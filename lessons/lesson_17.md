# Lesson 17: Fine-Tuning Pretrained Models

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain the difference between feature extraction and fine-tuning, and when each is appropriate
- Implement two-phase training: first training only the new head, then fine-tuning the backbone
- Apply gradual unfreezing strategies to safely adapt pretrained weights
- Choose an appropriate learning rate for fine-tuning (typically 10×–100× smaller than initial training)
- Recognize and prevent catastrophic forgetting during fine-tuning
- Monitor validation curves to detect divergence during fine-tuning
- Save and reload fine-tuned model checkpoints

---

## Detailed Explanation

### Feature Extraction vs. Fine-Tuning: The Key Difference

In Lesson 16, we used the pretrained backbone purely as a **frozen feature extractor**: the ImageNet weights never changed, and only the new classification head was trained. This is fast and effective, but it has a ceiling — the pretrained features are optimized for ImageNet, not your specific task.

**Fine-tuning** goes further. After the classification head has been trained to a reasonable performance level, you **unfreeze** some or all of the pretrained layers and continue training the entire (or partial) model — but with a much smaller learning rate. This allows the pretrained features to gently shift to better represent the patterns in your specific domain, without erasing the useful knowledge already encoded in the weights.

The analogy: feature extraction is like hiring an expert and putting them to work immediately on your problem. Fine-tuning is like giving that expert some additional on-the-job training tailored to your company's specific workflow. Both are better than training a fresh employee from scratch.

### Why Fine-Tune?

Pretrained models are trained on ImageNet, which contains everyday natural images — animals, vehicles, household objects, food. If your task involves images from a **different domain** — medical scans, satellite imagery, industrial defect detection, underwater photography — the higher-level features of the pretrained model may not be well-adapted to your domain. Fine-tuning allows the network to adapt those higher-level representations while retaining the low-level structure (edges, textures) that is nearly universal.

Fine-tuning is also beneficial even for natural-image tasks when your dataset is **large enough** to support it. With more data, the risk of overfitting the pretrained weights decreases, and the gains from fine-tuning become more consistent.

### When to Fine-Tune vs. Only Extract Features

| Scenario | Recommended Approach |
|---|---|
| Very small dataset (< 500 images), similar domain | Feature extraction only |
| Small-medium dataset (500–5000 images), similar domain | Feature extraction + optional fine-tuning of top layers |
| Medium dataset, different domain | Feature extraction + fine-tuning top 20–50% of layers |
| Large dataset (> 10,000 images per class) | Full fine-tuning or train from scratch |

### Catastrophic Forgetting: The Biggest Risk

When you unfreeze pretrained layers and train with a large learning rate, the carefully learned ImageNet representations can be **rapidly overwritten** by the gradients from your small dataset. This is called **catastrophic forgetting** — the model "forgets" everything it learned during pretraining and effectively regresses to a randomly initialized state.

The fix: always use a **very small learning rate** during fine-tuning. A typical rule of thumb:
- Initial head training: `learning_rate = 1e-3` (Adam default)
- Fine-tuning phase: `learning_rate = 1e-5` (100× smaller) or at most `1e-4` (10× smaller)

### The Two-Phase Training Recipe

The standard fine-tuning workflow follows exactly two phases:

**Phase 1 — Train the New Head (Feature Extraction)**

```
Epoch 1-10 (or until convergence):
  - Backbone weights: FROZEN (trainable=False)
  - Head weights: TRAINABLE
  - Learning rate: 1e-3 (Adam default)
  - Purpose: Initialize the head weights to reasonable values
```

This phase is critical. If you immediately unfreeze the backbone with a randomly initialized head, the large gradients from the untrained head will destroy the pretrained features.

**Phase 2 — Fine-Tune with Unfrozen Layers**

```
Epoch 11-20 (or N more epochs):
  - Top N backbone layers: UNFROZEN (trainable=True)
  - Lower backbone layers: FROZEN (trainable=False)
  - Head weights: TRAINABLE
  - Learning rate: 1e-5 (much smaller!)
  - Purpose: Gently adapt top features to your domain
```

```
ASCII Diagram: Two-Phase Training

PHASE 1:                     PHASE 2:
┌─────────────────┐          ┌─────────────────┐
│ FROZEN BACKBONE │          │ FROZEN (lower)  │
│  (all layers)   │          │  Conv Block 1   │
│                 │          │  Conv Block 2   │
│                 │          ├─────────────────┤
│                 │   ───►   │ UNFROZEN (upper)│
│                 │          │  Conv Block N-1 │
│                 │          │  Conv Block N   │
├─────────────────┤          ├─────────────────┤
│ TRAINABLE HEAD  │          │ TRAINABLE HEAD  │
│  GAP → Dense    │          │  GAP → Dense    │
│  → Softmax      │          │  → Softmax      │
└─────────────────┘          └─────────────────┘
  LR: 1e-3                     LR: 1e-5
```

### Gradual Unfreezing

Instead of jumping from all-frozen to partially unfrozen, you can use **gradual unfreezing**: unfreeze layers one block at a time, from the top of the network downward. This gives each set of layers time to adapt before the layers beneath them are also unfrozen.

```
Step 1: Freeze ALL backbone layers, train head only
Step 2: Unfreeze last 20 layers, train with LR=1e-5
Step 3: Unfreeze last 40 layers, train with LR=1e-5
Step 4: Unfreeze entire backbone, train with LR=1e-5
```

Each step adds more flexibility to the model. The advantage is a smoother, more controlled adaptation with less risk of catastrophic forgetting.

### Why Lower Layers Should Stay Frozen Longer

The hierarchy of CNN features maps directly to a hierarchy of generality:

- **Low-level layers** (edges, blobs, gradients): Almost universally applicable. Changing these is almost never beneficial and can be harmful.
- **Mid-level layers** (textures, object parts): Moderately task-specific. Occasionally beneficial to fine-tune for domain shift.
- **High-level layers** (semantic concepts): Highly task-specific. These are the most valuable to fine-tune when your domain differs from ImageNet.

Fine-tuning strategy: **always start from the top (most task-specific) and work downward**.

### Learning Rate for Fine-Tuning

The choice of learning rate is the most critical hyperparameter during fine-tuning:

- Too large: catastrophic forgetting — pretrained knowledge is destroyed
- Too small: no effective adaptation — fine-tuning has no benefit
- Just right: gentle, gradual adaptation of high-level features

Use an Adam optimizer with a reduced learning rate, or SGD with momentum. A decaying learning rate schedule (cosine decay or exponential decay) during fine-tuning can also help.

### Model Checkpoint Strategy

During fine-tuning, save checkpoints based on **validation accuracy or loss** using `ModelCheckpoint`. Because the model can sometimes diverge during fine-tuning (especially if the learning rate is too large), it is essential to be able to roll back to the best checkpoint.

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    'best_finetuned_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)
```

### Validation Monitoring: Watch for Divergence

During fine-tuning, monitor both training and validation loss closely. Warning signs of a problem:
- Validation loss **increases** while training loss decreases → overfitting
- Both losses **increase** → learning rate is too high (catastrophic forgetting)
- Validation loss oscillates wildly → learning rate is still too high

Early stopping (`EarlyStopping` callback) is your safety net:

```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)
```

### Layer-Wise Learning Rate Decay

An advanced technique is to assign different learning rates to different parts of the network: lower rates for lower layers (which should change very little) and higher rates for upper layers and the new head. This requires a custom optimizer setup or frameworks like `tensorflow_addons`. While beyond the scope of this lesson, it represents the state of the art in transfer learning practice.

---

## Key Concepts Table

| Concept | Definition |
|---|---|
| Fine-Tuning | Unfreezing pretrained layers and continuing training with a small LR |
| Two-Phase Training | Phase 1: train head only; Phase 2: unfreeze top layers and fine-tune |
| Catastrophic Forgetting | Rapid loss of pretrained knowledge when learning rate is too large |
| Gradual Unfreezing | Unfreezing layers progressively from top to bottom over multiple phases |
| Learning Rate for Fine-Tuning | Typically 10×–100× smaller than initial training (e.g., 1e-5 vs. 1e-3) |
| ModelCheckpoint | Keras callback to save the best model during training |
| EarlyStopping | Keras callback to halt training when a monitored metric stops improving |
| Layer-Wise LR Decay | Assigning smaller learning rates to lower layers during fine-tuning |
| Domain Shift | The difference between the distribution of training data and target data |
| Validation Divergence | A signal that fine-tuning is proceeding too aggressively |

---

## Code Reference

```python
import tensorflow as tf

# --- Phase 1: Feature Extraction ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3), include_top=False, weights='imagenet'
)
base_model.trainable = False  # Freeze backbone

inputs = tf.keras.Input(shape=(96, 96, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
model = tf.keras.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history1 = model.fit(train_ds, epochs=5, validation_data=val_ds)

# --- Phase 2: Fine-Tuning ---
base_model.trainable = True
# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # Much smaller LR!
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history2 = model.fit(train_ds, epochs=5, validation_data=val_ds)
```

---

## Activities

1. **Two-Phase Training Curves:** Starting from the frozen feature extractor (Phase 1) trained in Lesson 16, unfreeze the last 20 layers of MobileNetV2 and run Phase 2 fine-tuning for 10 epochs with `Adam(lr=1e-5)`. Plot training and validation accuracy for both phases on a single graph, annotating the boundary between phases.

2. **Learning Rate Sensitivity:** Run three separate Phase 2 fine-tuning runs with learning rates `[1e-3, 1e-4, 1e-5]`. Plot the validation loss for all three on the same axes. Print which learning rate causes the validation loss to spike (catastrophic forgetting) and which converges most smoothly.
## Review Questions

1. What is the key difference between feature extraction and fine-tuning?
2. Why must you complete Phase 1 (head training) before starting Phase 2 (fine-tuning)?
3. What is catastrophic forgetting and how does a small learning rate prevent it?
4. Why should lower-level layers remain frozen longer than upper-level layers during gradual unfreezing?
5. What callbacks would you add to a fine-tuning run to make it robust?
6. How would you decide which layers to unfreeze for a task involving satellite imagery (quite different from ImageNet)?
7. Why must you recompile the model after changing `base_model.trainable = True`?
8. What does validation divergence during fine-tuning typically indicate?

---

## Further Reading

- [Keras Fine-Tuning Guide](https://keras.io/guides/transfer_learning/#fine-tuning)
- [ULMFiT: Universal Language Model Fine-tuning (Gradual Unfreezing)](https://arxiv.org/abs/1801.06146)
- [An Overview of Catastrophic Forgetting](https://arxiv.org/abs/1612.00796)
- [Revisiting Unreasonable Effectiveness of Data](https://arxiv.org/abs/1707.02968)
- [How to Use Learning Rate Schedules in Keras](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules)
- [Practical Deep Learning for Coders – fast.ai](https://course.fast.ai/)
