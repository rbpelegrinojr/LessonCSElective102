# Lesson 16: Transfer Learning

## Learning Objectives
- Explain what transfer learning is and articulate why it is more efficient than training from scratch for most real-world tasks
- Describe how hierarchical feature learning in CNNs makes pretrained weights transferable across domains
- Identify the most popular pretrained architectures (VGG16, ResNet50, MobileNetV2, EfficientNet) and explain when to choose each
- Implement the feature extraction workflow: load a pretrained base, freeze its weights, and attach a custom classification head
- Evaluate the accuracy gap between a transfer-learning model and a from-scratch baseline on a small dataset
- Define bottleneck features and explain how they accelerate the training loop

## Detailed Explanation

### What Is Transfer Learning?

Imagine you already know how to ride a bicycle. When you decide to learn to ride a motorcycle, you do not start from zero — your balance skills, your understanding of steering, and your road awareness all carry over. You only need to learn what is new: throttle control, braking distances, and shifting gears. This is exactly what transfer learning does for neural networks.

Transfer learning is the practice of taking a model trained on one task (the **source task**) and reusing its learned representations to solve a different but related task (the **target task**). Instead of initialising all weights randomly and optimising them from scratch — a process that typically requires millions of labelled examples and days of GPU compute — you begin with weights that already encode rich, useful knowledge.

### Why Do Pretrained Weights Transfer?

The reason pretrained CNN weights generalise so well comes down to the hierarchical nature of convolutional feature learning:

```
Input Image
    │
    ▼
Layer 1 — Detects low-level features: edges, corners, colour gradients
    │
    ▼
Layer 2 — Combines edges into textures: stripes, grids, dots
    │
    ▼
Layer 3 — Combines textures into parts: eyes, wheels, leaves
    │
    ▼
Layer 4 — Combines parts into objects: faces, cars, trees
    │
    ▼
Layer 5 (classifier) — Task-specific: "cat" vs "dog" vs "car"
```

The early and middle layers learn features that are almost universally useful for any vision task — edges exist in medical scans, satellite imagery, and product photos alike. Only the final layers encode task-specific discriminative patterns. This is why we can take a network trained on ImageNet's 1.4 million photographs across 1,000 categories and repurpose the early layers for a completely new domain.

### ImageNet and Its Importance

**ImageNet** (specifically the ILSVRC benchmark) is a dataset containing roughly **1.4 million images** spanning **1,000 object categories** — from goldfish and vultures to espresso machines and parachutes. Since 2010, it has served as the standard benchmark for image classification research.

Training a high-accuracy model on ImageNet from scratch requires:
- Weeks of training on dozens of GPUs
- Access to all 1.4 million labelled images
- Careful architecture design and hyperparameter search

By downloading a model already trained on ImageNet, you inherit all of that effort for free. The pretrained weights represent a compressed summary of what an enormous vision model learned about the visual world.

### Popular Pretrained Architectures

| Architecture | Released | Top-5 Accuracy (ImageNet) | Parameters | Best Use Case |
|---|---|---|---|---|
| VGG16 | 2014 | 92.7 % | 138 M | Simple baseline, easy to visualise |
| ResNet50 | 2015 | 93.3 % | 25 M | General purpose, fast convergence |
| MobileNetV2 | 2018 | 91.0 % | 3.4 M | Mobile / edge devices, speed-critical |
| EfficientNetB0 | 2019 | 93.3 % | 5.3 M | Best accuracy/parameter trade-off |

**VGG16** stacks sixteen weight layers using only 3×3 convolutions — conceptually simple, computationally expensive, but excellent for understanding CNNs.

**ResNet50** introduces skip connections that allow gradients to flow directly through shortcut paths, solving the vanishing gradient problem and enabling 50+ layer networks.

**MobileNetV2** uses depthwise-separable convolutions to slash parameter counts, making it suitable for on-device inference on smartphones.

**EfficientNet** scales depth, width, and resolution simultaneously using a compound coefficient, achieving state-of-the-art efficiency.

### The Feature Extraction Workflow

Feature extraction is the simpler of the two transfer-learning strategies. The workflow is:

```
Step 1: Load pretrained model (e.g. MobileNetV2 trained on ImageNet)
         ┌─────────────────────────────────┐
         │  Pretrained Base (frozen)       │
         │  weights = ImageNet knowledge   │
         │  trainable = FALSE              │
         └─────────────────────────────────┘
                        │
Step 2: Remove original classification head (1000-class softmax)

Step 3: Add custom head for YOUR task
         ┌──────────────────────────────────┐
         │  GlobalAveragePooling2D          │
         │  Dense(256, activation='relu')   │
         │  Dropout(0.5)                    │
         │  Dense(10, activation='softmax') │  ← 10 CIFAR classes
         └──────────────────────────────────┘

Step 4: Train ONLY the new head (base layers frozen)
Step 5: Evaluate on test set
```

**Freezing** a layer means setting `layer.trainable = False`. During backpropagation, frozen layers do not update their weights — they act as a fixed feature extractor. Only the new head has `trainable = True`, so only those parameters change during gradient descent.

### Bottleneck Features

A powerful optimisation when doing feature extraction is to pre-compute **bottleneck features** — the output of the frozen base network for every training image — and save them to disk. Then you train only the small classification head using these pre-computed vectors, never passing data through the large base network during training. This can reduce training time by 10–50× because the expensive forward pass through hundreds of convolutional layers is done only once.

```
[Image] → [Frozen Base] → [Feature Vector 1280-dim] → saved to disk
                                                               │
Training loop only:    [Saved Feature Vectors] → [New Head] → loss → backprop
```

### When to Use Transfer Learning

Transfer learning is almost always the right choice when:
1. **Your dataset is small** (fewer than ~50,000 labelled examples) — training from scratch would overfit badly.
2. **Your domain is similar to ImageNet** — natural photographs, animals, objects, scenes.
3. **You have limited compute** — GPUs or training time are constrained.
4. **You need a quick baseline** — pretrained models converge in minutes to hours rather than days.

It may be less beneficial when:
- Your data is very unlike ImageNet (e.g. raw X-ray pixels, sonar waveforms, microscopy images) — though even then, early layers often still generalise.
- You have millions of domain-specific labelled examples.

### Domain Similarity

Think of domain similarity on a spectrum:

```
Very Similar to ImageNet          Less Similar to ImageNet
────────────────────────────────────────────────────────▶
Pet photos  | Food photos | Satellite | Medical scans | Sonar
(reuse all) | (reuse most)| (reuse early)| (reuse early) | (experiment)
```

The more similar the target domain is to ImageNet, the more layers you can safely reuse with high confidence.

### Common Misconceptions

**"Pretrained models are always better."** Not necessarily. If you have millions of domain-specific images very different from ImageNet, training from scratch with a well-tuned architecture can sometimes outperform a pretrained model.

**"I should always use the biggest pretrained model."** Bigger is not always better. EfficientNetB0 often outperforms VGG16 while using 26× fewer parameters. Choose based on your accuracy/latency/memory requirements.

**"Freezing more layers is always safer."** Freezing all base layers may leave performance on the table. Lesson 17 covers the fine-tuning strategy for unlocking additional accuracy.

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| Transfer Learning | Reusing a model trained on one task to improve learning on a different task | Reduces data and compute requirements dramatically |
| Pretrained Model | A neural network whose weights were already optimised on a large dataset | Provides a head-start encoding general visual knowledge |
| Feature Extraction | Using a frozen pretrained base to generate fixed feature vectors for a new task | Fastest and safest form of transfer learning |
| Bottleneck Features | Pre-computed activations from the frozen base, saved before the classifier | Speeds up training by 10–50× by avoiding repeated forward passes |
| Frozen Layer | A layer with `trainable = False`; its weights do not change during training | Preserves valuable pretrained knowledge |
| ImageNet | 1.4M image, 1000-class benchmark dataset used to train most popular CNN bases | De-facto standard for pretraining vision models |
| Domain Similarity | How closely the distribution of the target dataset matches the source training data | Determines how many pretrained layers can be safely reused |
| Fine-Tuning | Unfreezing some pretrained layers and training them at a low learning rate | Covered in Lesson 17; further improves accuracy after feature extraction |
| VGG16 / ResNet50 | Two classic pretrained architectures from the ImageNet era | Common baselines; available in Keras Applications |
| MobileNetV2 | Lightweight architecture using depthwise-separable convolutions | Ideal for mobile and edge deployment |

## Code Reference

See the fully runnable demonstration in **`code/lesson_16.py`**.

## Activities

1. **Explore Architecture Shapes** — Load VGG16 and MobileNetV2 using `keras.applications` with `include_top=False`. Print `model.summary()` for both. Count the total parameters in each base and explain why MobileNetV2 is more efficient.

2. **Build a Custom Head** — Attach a new classification head (GlobalAveragePooling → Dense 256 ReLU → Dropout 0.5 → Dense 10 softmax) to MobileNetV2. Verify that the total trainable parameters match only the head, not the base.

3. **Feature Extraction Training** — Train your feature-extraction model on CIFAR-10 for 10 epochs. Record train/validation accuracy and loss at each epoch. Plot learning curves using Matplotlib.

4. **Baseline Comparison** — Train a simple 4-layer CNN from scratch on CIFAR-10 with the same number of training images. Compare final test accuracy against the transfer-learning model. Write a one-paragraph explanation of the difference.

5. **Bottleneck Feature Experiment** — Use the frozen MobileNetV2 base to pre-compute feature vectors for all 50,000 CIFAR-10 training images and save them using `numpy.save()`. Then train a small MLP on those vectors and compare training time to the standard pipeline.

## Review Questions

1. Why are the early convolutional layers of an ImageNet-trained model useful even for tasks very different from recognising everyday objects?
2. What does it mean to "freeze" a layer, and what practical effect does this have on the backpropagation algorithm?
3. You have 3,000 labelled images of plant diseases. Would you use transfer learning or train from scratch? Justify your answer with reference to dataset size and domain similarity.
4. Explain in your own words what bottleneck features are and why pre-computing them can speed up the training loop.
5. Compare VGG16 and MobileNetV2 in terms of parameter count, accuracy, and use case. When would you choose MobileNetV2 over VGG16?

## Further Reading

- **"ImageNet Large Scale Visual Recognition Challenge"** — Russakovsky et al. (2015) — the paper that defined the ImageNet benchmark and inspired the deep-learning revolution in computer vision
- **"Deep Learning" by Goodfellow, Bengio & Courville — Chapter 15: Representation Learning** — the theoretical foundation for why learned representations transfer
- **Keras Applications Documentation** — `https://keras.io/api/applications/` — full list of pretrained models with accuracy benchmarks and usage examples
- **"How transferable are features in deep neural networks?"** — Yosinski et al. (2014) — empirical study of exactly which layers transfer across tasks
- **"EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"** — Tan & Le (2019) — introduces compound scaling and the EfficientNet family
