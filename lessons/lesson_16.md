# Lesson 16: Transfer Learning

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain what transfer learning is and why it is effective for image classification tasks
- Distinguish between **feature extraction** and **fine-tuning** as two distinct modes of transfer learning
- Load and use popular pretrained models available in Keras (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNet)
- Freeze and unfreeze layers using `trainable = False` and `trainable = True`
- Build a custom classification head on top of a pretrained backbone
- Recognize common mistakes when applying transfer learning, such as improper input normalization

---

## Detailed Explanation

### What Is Transfer Learning?

Transfer learning is the practice of taking a model that has already been trained on one task and reusing it — either in part or in full — to help solve a different but related task. Rather than starting from scratch and randomly initializing every weight, you begin with weights that already encode meaningful knowledge. In the world of computer vision, this typically means loading a model that was trained on a large, general-purpose image dataset and adapting it to your own, often much smaller, dataset.

The analogy in human learning is intuitive: a person who already knows how to ride a bicycle has a much easier time learning to ride a motorcycle than someone who has never balanced on two wheels. Prior knowledge transfers.

### Why Does Transfer Learning Work?

The reason transfer learning is so powerful for convolutional neural networks lies in what the different layers of a CNN actually learn. Researchers have shown through visualization studies that the **early layers** of a deep CNN — regardless of the task it was trained on — tend to learn very generic, low-level features:

- **Layer 1–2**: Oriented edges, color blobs, simple gradients
- **Layer 3–4**: Textures, corners, repeating patterns
- **Layer 5–6**: Object parts (wheels, eyes, leaves)
- **Later layers**: High-level semantic representations specific to the training classes

Because early-layer features are universal and task-agnostic, they are directly applicable to almost any image recognition problem. A network trained to distinguish cats from dogs has already learned to detect fur texture, eye shapes, and curved lines — all of which are also useful for recognizing flowers or classifying X-rays.

### ImageNet: The Source of Most Pretrained Weights

The most commonly used source of pretrained weights is **ImageNet**, a massive dataset containing over 14 million labeled images spanning 1,000 categories (cars, animals, household objects, food, etc.). Training a full model on ImageNet from scratch requires weeks of compute on dozens of GPUs. Fortunately, the deep learning community has made the resulting weights publicly available, so you can download a fully trained model in seconds.

When you call `tf.keras.applications.MobileNetV2(weights='imagenet')`, Keras downloads weights that encode over 1.2 million training examples of 1,000 real-world categories. Your task likely has far fewer images and far fewer classes, but you still get the benefit of all that representational learning.

### Feature Extraction vs. Fine-Tuning

There are two primary ways to apply transfer learning:

**1. Feature Extraction**

In feature extraction, you keep the pretrained model's weights completely frozen (i.e., they do not change during training) and use the model purely as a fixed feature extractor. You then attach a new classification head — typically a `GlobalAveragePooling2D` layer followed by one or more `Dense` layers — and train only this new head.

This approach is the safer, faster option and is ideal when:
- Your dataset is **small** (fewer than a few thousand images)
- Your domain is **similar** to ImageNet (natural photos, everyday objects)
- Compute resources are limited

Because the backbone weights never change, backpropagation only flows through the small new head, making each training step very fast.

**2. Fine-Tuning**

Fine-tuning goes a step further: after training the new head, you unfreeze some or all of the pretrained layers and continue training the entire model — or at least the top portion — with a very small learning rate. This allows the pretrained features to be slightly adjusted to better match your specific domain.

Fine-tuning is covered in detail in Lesson 17. For this lesson, we focus on feature extraction.

### When Transfer Learning Is Most Valuable

Transfer learning delivers its biggest benefits when you have a **small dataset**. Training a deep CNN from scratch typically requires tens of thousands of labeled images to generalize well. With transfer learning, you can achieve excellent results with as few as **100–500 images per class**, because the pretrained backbone already provides rich, general-purpose visual representations.

Expected gains:
- Reach **90% accuracy** on a small dataset with ~100 labeled examples per class, where training from scratch might require 10,000+ examples to reach the same performance
- Dramatically **shorter training time**: instead of training for 50+ epochs, 5–10 epochs on the new head is often sufficient
- **Better generalization**: pretrained features act as a powerful regularizer

### Popular Pretrained Models in Keras

Keras provides many pretrained architectures via `tf.keras.applications`:

| Model | Parameters | ImageNet Top-1 Accuracy | Best Use Case |
|---|---|---|---|
| VGG16 | 138M | 71.3% | Simple baseline, easy to visualize |
| ResNet50 | 25M | 74.9% | General purpose, good balance |
| MobileNetV2 | 3.4M | 71.8% | Mobile/edge deployment |
| InceptionV3 | 23M | 77.9% | High accuracy, moderate size |
| EfficientNetB0 | 5.3M | 77.1% | Best accuracy/parameter ratio |

For educational purposes, **MobileNetV2** is an excellent choice because it is fast to load, small in memory, and still highly accurate.

### Model Architecture: Backbone + Classification Head

Every pretrained model used for transfer learning is composed of two logical parts:

```
┌──────────────────────────────────────────────────┐
│           PRETRAINED BACKBONE (Frozen)            │
│  Conv Block 1 → Conv Block 2 → ... → Conv Block N │
│        (Edges) → (Textures) → (Object Parts)      │
└──────────────────────────┬───────────────────────┘
                           │
                           ▼
               [Feature Maps: e.g., 7×7×1280]
                           │
                           ▼
┌──────────────────────────────────────────────────┐
│          NEW CLASSIFICATION HEAD (Trained)        │
│    GlobalAveragePooling2D → Dense(256, ReLU)      │
│           → Dropout(0.5) → Dense(N, Softmax)      │
└──────────────────────────────────────────────────┘
                           │
                           ▼
              [Class Probabilities: N classes]
```

### The New Classification Head

The standard classification head consists of:

1. **GlobalAveragePooling2D**: Collapses the spatial dimensions of the feature maps by averaging, producing a single vector per feature map. This is more efficient and less prone to overfitting than `Flatten`.
2. **Dense layer(s)**: One or two fully connected layers with ReLU activation to learn task-specific combinations of features.
3. **Dropout**: Regularization to prevent overfitting on small datasets.
4. **Output Dense layer**: One unit per class with `softmax` activation for multi-class problems.

### Freezing and Unfreezing Layers

Setting `base_model.trainable = False` freezes all layers in the backbone. Setting it to `True` (or selectively setting `layer.trainable = True`) unfreezes them. After changing `trainable` status, **you must recompile the model** for the change to take effect.

```python
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False)
base_model.trainable = False  # Freeze all pretrained layers
```

### Domain Similarity and Its Effect

Transfer learning works best when the **source domain** (ImageNet) is similar to the **target domain** (your task). When transferring from ImageNet to medical images (e.g., chest X-rays), the early-layer features (edges, textures) still transfer well, but higher-level features may not. In such cases, fine-tuning more layers becomes more important.

### Critical Mistake: Input Normalization

Each pretrained model was trained with a specific input preprocessing scheme. If you feed raw pixel values (0–255) to a model that expects values preprocessed with `imagenet` statistics, your results will be poor. Always use the model's companion preprocessing function:

```python
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
x = preprocess_input(raw_image)  # Scales to [-1, 1] for MobileNetV2
```

---

## Key Concepts Table

| Concept | Definition |
|---|---|
| Transfer Learning | Reusing a model trained on one task for a different but related task |
| Pretrained Model | A model whose weights were learned from a large dataset (e.g., ImageNet) |
| Feature Extraction | Using frozen pretrained weights as a fixed feature extractor |
| Fine-Tuning | Unfreezing pretrained layers and continuing training with a small LR |
| Backbone | The convolutional base of a pretrained model (excludes the classification head) |
| Classification Head | The new layers added on top of the backbone for the target task |
| GlobalAveragePooling2D | Spatially averages feature maps into a single vector |
| Freezing | Setting `trainable = False` so weights do not update during training |
| ImageNet | 14M image, 1000-class dataset used to train most public CNN weights |
| Catastrophic Forgetting | Rapid overwriting of pretrained knowledge when learning rate is too high |

---

## Code Reference

```python
import tensorflow as tf

# Load MobileNetV2 without the top classification layer
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,       # Remove ImageNet classifier
    weights='imagenet'       # Load pretrained ImageNet weights
)
base_model.trainable = False  # Freeze the backbone

# Build new classification head
inputs = tf.keras.Input(shape=(96, 96, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

---

## Activities

1. **Explore Model Architectures**: Load VGG16, ResNet50, and MobileNetV2 with `include_top=False`. Print the number of layers and parameters for each. Which is largest? Which is smallest?

2. **Feature Extraction on CIFAR-10**: Use MobileNetV2 as a frozen feature extractor on CIFAR-10. Resize images to 96×96. Train only the new head for 5 epochs. Record the final validation accuracy.

3. **Compare Preprocessing**: Load a sample CIFAR-10 image and apply the MobileNetV2 `preprocess_input` function. Plot the original image and the preprocessed image side by side. What changed?

4. **Layer Inspector**: After loading MobileNetV2, iterate through `model.layers` and print the name, type, and `trainable` status of every layer.

5. **Experiment**: Change the number of units in the Dense head (64, 128, 256, 512). How does this affect validation accuracy and training time?

---

## Review Questions

1. What is transfer learning, and why is it particularly useful when you have a small dataset?
2. Explain the difference between feature extraction and fine-tuning.
3. Why do early layers of a CNN tend to transfer well across different tasks?
4. What is `include_top=False` doing when you load a pretrained model?
5. Why must you recompile the model after changing `layer.trainable`?
6. What is `GlobalAveragePooling2D` and why is it preferred over `Flatten` in transfer learning heads?
7. What could go wrong if you forget to apply the pretrained model's `preprocess_input` function?
8. Name three pretrained models available in Keras and describe a use case for each.

---

## Further Reading

- [Keras Transfer Learning Guide](https://keras.io/guides/transfer_learning/)
- [CS231n: Transfer Learning](https://cs231n.github.io/transfer-learning/)
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://image-net.org/challenges/LSVRC/)
- [A Survey on Transfer Learning – Pan & Yang (2010)](https://ieeexplore.ieee.org/document/5288526)
- [MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)
- [How transferable are features in deep neural networks? – Yosinski et al.](https://arxiv.org/abs/1411.1792)
