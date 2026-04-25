# Lesson 17: Fine-Tuning Pretrained Models

## Learning Objectives
- Distinguish between feature extraction and fine-tuning, and explain when each strategy is appropriate
- Implement a gradual unfreezing strategy that safely unlocks pretrained layers without destroying learned weights
- Apply discriminative learning rates (different rates for different layer groups) to stabilise fine-tuning
- Explain catastrophic forgetting and describe practical techniques to minimise its impact
- Evaluate model performance at each stage of fine-tuning and interpret the results on validation curves
- Recognise when fine-tuning helps versus when it hurts model performance

## Detailed Explanation

### Recap: Where Feature Extraction Ends

In Lesson 16 we froze the entire pretrained base and trained only a small classification head. This approach is fast and safe, and it works very well when your dataset is small and closely resembles ImageNet. But it leaves performance on the table: the frozen convolutional filters were optimised for 1,000 ImageNet categories, not for your specific ten or hundred categories. Fine-tuning is the next step that unlocks additional accuracy by teaching the pretrained layers to adapt to your exact task.

### Feature Extraction vs Fine-Tuning

```
Feature Extraction:                    Fine-Tuning:
┌──────────────────────┐               ┌──────────────────────┐
│  Base (ALL FROZEN)   │               │  Base (partially      │
│  trainable = False   │               │  UNFROZEN)            │
│  ← no gradient flow  │               │  trainable = True     │
├──────────────────────┤               │  ← gradient flows     │
│  Custom Head         │               │  (very low LR)        │
│  trainable = True    │               ├──────────────────────┤
│  ← gradient flows    │               │  Custom Head         │
│  (normal LR)         │               │  trainable = True    │
└──────────────────────┘               │  (normal or low LR)  │
                                       └──────────────────────┘
```

The critical difference: in fine-tuning, gradient updates flow back through some or all of the pretrained base, nudging those filters to specialise for your new task. Done carelessly, this can erase the valuable ImageNet knowledge. Done correctly, it produces a significantly better model.

### When to Fine-Tune

Fine-tuning is beneficial when:
- **You have enough data** (typically 10,000+ examples per class) to provide meaningful gradients without overfitting
- **Your domain differs from ImageNet** — medical images, satellite photographs, microscopy — where the mid-to-high level features of ImageNet networks do not perfectly match your data distribution
- **You have already reached the ceiling with feature extraction** and need a few more percentage points of accuracy
- **You can afford longer training** — fine-tuning always takes more compute than feature extraction

Fine-tuning may hurt when:
- Your dataset is tiny (< 1,000 examples) — the network will overfit to noise
- Your domain is very similar to ImageNet — there is little to gain and much to lose
- You use too high a learning rate — catastrophic forgetting will occur

### The Gradual Unfreezing Strategy

Never unfreeze all layers at once, especially at a high learning rate. The recommended procedure is:

```
Stage 1 — Train only the head (feature extraction)
          Epochs 1–10: LR = 1e-3, base frozen
          → validate: reach a reasonable starting accuracy

Stage 2 — Unfreeze the top block of the base
          Epochs 11–20: LR = 1e-4, only top N layers trainable
          → fine-tune high-level ImageNet features toward your task

Stage 3 — Unfreeze more blocks (optional)
          Epochs 21–30: LR = 1e-5, more layers trainable
          → deeper refinement; watch for overfitting

Stage 4 — Unfreeze all layers (optional, large datasets only)
          Epochs 31–40: LR = 1e-5 or lower
          → full-network fine-tuning
```

The intuition is that early layers (edge detectors) are already close to optimal for any vision task and need only tiny adjustments. Later layers (object-part detectors) are more task-specific and can benefit from larger updates — but even they should use learning rates 10× to 100× smaller than normal.

### Discriminative Learning Rates

A powerful refinement is to assign **different learning rates to different layer groups**. Lower layers (closer to the input) receive smaller learning rates; upper layers receive larger rates. This reflects the fact that lower-layer features are already nearly universal while upper-layer features are more task-specific.

```
Learning Rate Schedule Across Layers:
─────────────────────────────────────────────────────
Input  →  Block 1  →  Block 2  →  Block 3  →  Head
          LR=1e-5     LR=1e-4     LR=1e-4    LR=1e-3
─────────────────────────────────────────────────────
Smallest LR                              Largest LR
(most universal features)         (most task-specific)
```

In Keras, you can achieve this by using multiple optimiser parameter groups or by setting per-layer learning rate multipliers via custom training loops.

### Catastrophic Forgetting

**Catastrophic forgetting** (also called catastrophic interference) is the phenomenon where training on new data causes a neural network to abruptly lose performance on previously learned tasks or examples. In the context of fine-tuning, it manifests as the pretrained ImageNet knowledge being overwritten by the gradients from your small target dataset.

**Symptoms to watch for:**
- Validation loss suddenly spikes upward after unfreezing
- Train accuracy rises but validation accuracy falls (severe overfitting)
- The model starts predicting the same class for almost all inputs

**Mitigation strategies:**

1. **Use a very low learning rate** — the most important technique. A learning rate of 1e-4 to 1e-5 for unfrozen pretrained layers makes updates small enough to refine without destroying.

2. **Unfreeze gradually** — start with only the last block, train to convergence, then move deeper.

3. **Apply strong regularisation** — Dropout (0.3–0.5), L2 weight decay, and data augmentation all constrain updates.

4. **Use learning rate warmup** — ramp the learning rate from 0 to the target over the first epoch, preventing large updates at the start.

5. **Monitor validation metrics at every epoch** — if validation loss starts rising, reduce the learning rate or re-freeze layers.

### Progressive Resizing

A technique popularised by fast.ai: start training with small input images (e.g. 64×64), then gradually increase the resolution (128×128, then 224×224) across training stages. Benefits include:
- Faster early epochs (smaller images are cheaper to process)
- Built-in regularisation (the network must generalise across scales)
- Better final accuracy on high-resolution images

```
Stage 1: 64×64  → quick exploration, get weights roughly right
Stage 2: 128×128 → medium-resolution refinement
Stage 3: 224×224 → full-resolution fine-tuning
```

### BatchNorm Behaviour During Fine-Tuning

Batch Normalisation layers have their own trainable parameters (scale γ and shift β) as well as running statistics (mean and variance) accumulated during ImageNet training. During fine-tuning there are two choices:

- **Keep BatchNorm frozen** (`layer.trainable = False` even for BN layers, or call the base with `training=False`): uses ImageNet statistics, avoids disruption from small batches.
- **Allow BatchNorm to update**: can adapt statistics to the new domain but risks instability with small batch sizes.

The common recommendation is to keep BatchNorm layers frozen (in inference mode) during fine-tuning, especially with small datasets or small batch sizes.

### Practical Fine-Tuning Checklist

```
✅ Complete feature extraction first (converge the head)
✅ Save the feature-extraction checkpoint before unfreezing
✅ Reduce learning rate by 10× before unfreezing
✅ Unfreeze top block only → train → evaluate
✅ Apply data augmentation: flip, crop, colour jitter
✅ Use early stopping with patience ≥ 5
✅ Monitor both train and validation curves
✅ Compare against feature-extraction baseline before declaring success
```

### How Much Improvement to Expect

On typical datasets with 10k–100k images similar to ImageNet, fine-tuning the top block adds **1–5 percentage points** of accuracy over feature extraction alone. Fine-tuning all layers can add another **1–3 points** on top of that. The gains are real but modest — don't expect miracles. If you need large gains, the answer is usually more data, not deeper fine-tuning.

### A Note on Learning Rate Finders

Before fine-tuning, it is worth running a **learning rate range test** (LR finder): start with a very small learning rate (1e-7) and gradually increase it over several mini-batches, plotting loss versus learning rate. The optimal learning rate is just before the point where loss starts rising. This empirical technique, introduced by Leslie Smith (2017), removes guesswork from learning rate selection.

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| Fine-Tuning | Unfreezing pretrained layers and continuing training with a low learning rate | Adapts pretrained features to the specific target task for higher accuracy |
| Gradual Unfreezing | Unfreezing layer groups one at a time from top to bottom | Prevents catastrophic forgetting by making small, stable updates |
| Catastrophic Forgetting | Loss of previously learned knowledge when training on new data | The primary risk in fine-tuning; requires low LR and gradual unfreezing |
| Discriminative Learning Rates | Assigning different learning rates to different layer groups | Allows fast adaptation in upper layers while preserving universal lower-layer features |
| Progressive Resizing | Training at increasing image resolutions across stages | Speeds up training and improves generalisation |
| Learning Rate Finder | Empirical test to find the optimal learning rate before fine-tuning | Eliminates guesswork; prevents training instability |
| BatchNorm in Fine-Tuning | Keeping BN layers frozen preserves stable running statistics | Avoids instability with small batches during fine-tuning |
| Feature Extraction Ceiling | Maximum accuracy achievable with a frozen base | The starting point from which fine-tuning seeks improvement |
| Layer Groups | Logical groupings of layers (blocks) in a pretrained model | Used to apply discriminative LRs and gradual unfreezing |
| Early Stopping | Halting training when validation metric stops improving | Prevents overfitting; saves the best checkpoint automatically |

## Code Reference

See the fully runnable demonstration in **`code/lesson_17.py`**.

## Activities

1. **Stage-by-Stage Accuracy Table** — Starting from the feature-extraction checkpoint of Lesson 16, create a table tracking validation accuracy after: (a) head-only training, (b) unfreezing the top block, and (c) unfreezing the top two blocks. Plot accuracy vs epoch for all three stages on one graph.

2. **Catastrophic Forgetting Experiment** — Deliberately induce catastrophic forgetting: unfreeze all layers at once and train with a high learning rate (1e-2). Plot the validation loss curve. Explain what you observe and compare it to the gradual-unfreezing curve.

3. **Discriminative Learning Rate Implementation** — Manually split the MobileNetV2 base into three groups (bottom third, middle third, top third). Assign learning rates 1e-5, 1e-4, and 1e-3 to each group respectively, plus 1e-3 for the head. Train for 10 epochs and compare to uniform fine-tuning.

4. **Learning Rate Finder** — Implement a simple LR finder: train for one epoch while increasing the LR exponentially from 1e-7 to 1e-1. Plot loss vs LR and identify the optimal range.

5. **Progressive Resizing** — Modify the data pipeline to first train at 64×64 input resolution, then 128×128, then 224×224. Compare final accuracy and total training time against training exclusively at 224×224 from the start.

## Review Questions

1. What is the main risk of unfreezing all pretrained layers simultaneously and training with a high learning rate? How does gradual unfreezing address this risk?
2. You have 100,000 photographs of industrial defects that look quite different from everyday ImageNet images. Should you use feature extraction, fine-tuning, or train from scratch? Justify your answer.
3. Explain why BatchNorm layers are typically kept in inference mode during fine-tuning, especially with small batch sizes.
4. Describe the learning rate finder technique and explain why it is useful before starting fine-tuning.
5. After unfreezing layers and running 5 epochs, you notice validation accuracy is decreasing while training accuracy continues to rise. What has likely gone wrong, and what steps would you take to fix it?

## Further Reading

- **"Revisiting Unreasonable Effectiveness of Data"** — Sun et al. (2017) — shows how dataset size affects fine-tuning benefit; useful context for deciding when to fine-tune
- **"Universal Language Model Fine-Tuning for Text Classification" (ULMFiT)** — Howard & Ruder (2018) — introduced gradual unfreezing and discriminative learning rates in NLP, applicable to vision
- **fast.ai Practical Deep Learning for Coders — Lesson 5** — practical walkthrough of fine-tuning with progressive resizing using real-world datasets
- **"A Disciplined Approach to Neural Network Hyper-Parameters"** — Smith (2018) — describes the LR finder and cyclical learning rates, tools essential for fine-tuning
- **Keras Transfer Learning & Fine-Tuning Tutorial** — `https://keras.io/guides/transfer_learning/` — official Keras guide with runnable examples
