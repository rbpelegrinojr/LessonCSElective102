"""
Lesson 14: Data Augmentation
==============================
This module demonstrates data augmentation techniques that improve generalization
by teaching CNNs to learn invariant features. Topics covered:
  - Visualizing standard geometric and color augmentations
  - Using ImageDataGenerator (legacy Keras augmentation API)
  - Using Keras Preprocessing Layers (modern GPU-based API)
  - Implementing MixUp augmentation from scratch
  - Comparing model performance with and without augmentation on CIFAR-10
"""

# === Standard library and framework imports ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("Lesson 14: Data Augmentation")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# CIFAR-10 class names
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ===========================================================
# === SECTION 1: VISUALIZING AUGMENTATION TECHNIQUES ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 1: Visualizing Augmentation Techniques")
print("=" * 60)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()

# Normalize to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

# Pick a sample image for augmentation visualization
sample_idx  = 7
sample_img  = x_train[sample_idx]                    # Shape: (32, 32, 3)
sample_label = CIFAR10_CLASSES[y_train[sample_idx]]
print(f"Sample image class: '{sample_label}'")

# Build an augmentation generator with many different transformations
augmentation_configs = {
    "Original":             ImageDataGenerator(),
    "Horizontal Flip":      ImageDataGenerator(horizontal_flip=True),
    "Vertical Flip":        ImageDataGenerator(vertical_flip=True),
    "Rotation ±30°":        ImageDataGenerator(rotation_range=30),
    "Rotation ±90°":        ImageDataGenerator(rotation_range=90),
    "Zoom In (0.8-1.0)":    ImageDataGenerator(zoom_range=[0.8, 1.0]),
    "Zoom Out (1.0-1.3)":   ImageDataGenerator(zoom_range=[1.0, 1.3]),
    "Width Shift 20%":      ImageDataGenerator(width_shift_range=0.2),
    "Height Shift 20%":     ImageDataGenerator(height_shift_range=0.2),
    "Brightness +30%":      ImageDataGenerator(brightness_range=[1.2, 1.5]),
    "Brightness -30%":      ImageDataGenerator(brightness_range=[0.5, 0.8]),
    "Shear":                ImageDataGenerator(shear_range=20),
}

# Set random seeds for reproducible augmentation examples
np.random.seed(42)
tf.random.set_seed(42)

fig, axes = plt.subplots(3, 4, figsize=(16, 12))
fig.suptitle(f"Lesson 14: Augmentation Gallery — Class: '{sample_label}'\n"
             f"(All augmented images retain the same label)",
             fontsize=13, fontweight='bold')
axes_flat = axes.flatten()

# Apply each augmentation and display result
for ax, (aug_name, aug_gen) in zip(axes_flat, augmentation_configs.items()):
    # flow() expects a batch: shape (1, H, W, C)
    batch = np.expand_dims(sample_img, 0)
    aug_gen.fit(batch)
    # Get one augmented sample
    aug_batch = next(aug_gen.flow(batch, batch_size=1))
    aug_img = np.clip(aug_batch[0], 0.0, 1.0)   # Clip to valid pixel range

    ax.imshow(aug_img)
    ax.set_title(aug_name, fontsize=9, fontweight='bold')
    ax.axis('off')

plt.tight_layout()
plt.savefig("lesson_14_augmentation_gallery.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_14_augmentation_gallery.png")
print("  Observation: All 12 images show the same object with different transforms —")
print("  the label stays the same, but the model sees more variation.")


# ===========================================================
# === SECTION 2: KERAS PREPROCESSING LAYERS (MODERN API) ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 2: Keras Preprocessing Layers (GPU-based Augmentation)")
print("=" * 60)

# Modern way: augmentation as part of the model graph using tf.keras.layers
# These run on the GPU and are automatically disabled during inference
augmentation_pipeline = keras.Sequential([
    layers.RandomFlip("horizontal"),             # Flip left-right randomly
    layers.RandomRotation(0.15),                 # Rotate by up to ±15% of 2π radians
    layers.RandomZoom(0.15),                     # Zoom in/out by up to 15%
    layers.RandomTranslation(0.1, 0.1),          # Translate by up to 10% of height/width
    layers.RandomContrast(0.2),                  # Adjust contrast by up to ±20%
], name="augmentation_pipeline")

print("Augmentation pipeline:")
augmentation_pipeline.summary()

# Visualize augmentation pipeline output on multiple passes
fig, axes = plt.subplots(4, 6, figsize=(16, 12))
fig.suptitle("Lesson 14: Keras Preprocessing Layer Augmentation\n"
             "(Same image, 24 different augmented versions)",
             fontsize=13, fontweight='bold')

sample_tensor = tf.convert_to_tensor(
    np.expand_dims(sample_img, 0)               # Shape: (1, 32, 32, 3)
)

for i, ax in enumerate(axes.flatten()):
    aug_out = augmentation_pipeline(sample_tensor, training=True)  # training=True enables augmentation
    aug_img = np.clip(aug_out[0].numpy(), 0.0, 1.0)
    ax.imshow(aug_img)
    ax.set_title(f"v{i+1}", fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.savefig("lesson_14_keras_augmentation_layers.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_14_keras_augmentation_layers.png")


# ===========================================================
# === SECTION 3: MIXUP AUGMENTATION FROM SCRATCH ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 3: MixUp Augmentation")
print("=" * 60)

def mixup_batch(x_batch, y_batch, alpha=0.4):
    """
    MixUp augmentation: blend two random images and their labels.
    
    For each sample in the batch, a mixing coefficient lambda is drawn from
    Beta(alpha, alpha) distribution. The blended sample is:
        x_mix = lambda * x_i + (1 - lambda) * x_j
        y_mix = lambda * y_i + (1 - lambda) * y_j
    
    Returns:
        x_mix: blended images
        y_mix: blended one-hot labels
    """
    batch_size = x_batch.shape[0]
    n_classes  = 10

    # Convert integer labels to one-hot for soft label blending
    y_onehot = tf.one_hot(y_batch, n_classes).numpy()

    # Draw lambda from Beta(alpha, alpha) — values near 0.5 give equal mix
    lambdas = np.random.beta(alpha, alpha, size=batch_size)
    lambdas = np.maximum(lambdas, 1 - lambdas)    # Ensure lambda >= 0.5 (convention)

    # Create a shuffled permutation for the second sample in each pair
    idx_shuffle = np.random.permutation(batch_size)

    x_mix = np.zeros_like(x_batch)
    y_mix = np.zeros_like(y_onehot)

    for i in range(batch_size):
        lam = lambdas[i]
        j   = idx_shuffle[i]
        x_mix[i] = lam * x_batch[i] + (1 - lam) * x_batch[j]   # Pixel-level blend
        y_mix[i] = lam * y_onehot[i] + (1 - lam) * y_onehot[j] # Label blend

    return x_mix.astype("float32"), y_mix.astype("float32")

# Visualize MixUp examples
n_examples = 6
indices_a = np.random.choice(len(x_train), n_examples, replace=False)
indices_b = np.random.choice(len(x_train), n_examples, replace=False)
lambdas   = np.random.beta(0.4, 0.4, n_examples)
lambdas   = np.maximum(lambdas, 1 - lambdas)

fig, axes = plt.subplots(n_examples, 3, figsize=(9, 18))
fig.suptitle("Lesson 14: MixUp Augmentation\n(Image A + Image B → Mixed Image with blended labels)",
             fontsize=12, fontweight='bold')

for row in range(n_examples):
    ia, ib = indices_a[row], indices_b[row]
    lam    = lambdas[row]
    mixed  = np.clip(lam * x_train[ia] + (1 - lam) * x_train[ib], 0, 1)

    label_a = CIFAR10_CLASSES[y_train[ia]]
    label_b = CIFAR10_CLASSES[y_train[ib]]

    axes[row, 0].imshow(x_train[ia]); axes[row, 0].axis('off')
    axes[row, 0].set_title(f"Image A\n'{label_a}'", fontsize=8)

    axes[row, 1].imshow(mixed); axes[row, 1].axis('off')
    axes[row, 1].set_title(f"Mixed (λ={lam:.2f})\n{lam:.0%}×'{label_a}' + {1-lam:.0%}×'{label_b}'",
                           fontsize=7)

    axes[row, 2].imshow(x_train[ib]); axes[row, 2].axis('off')
    axes[row, 2].set_title(f"Image B\n'{label_b}'", fontsize=8)

plt.tight_layout()
plt.savefig("lesson_14_mixup_examples.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_14_mixup_examples.png")
print("  Key insight: The model must predict a mixture of labels — this acts as")
print("  strong regularization and improves probability calibration.")


# ===========================================================
# === SECTION 4: COMPARING TRAINING WITH AND WITHOUT AUGMENTATION ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 4: Training With vs. Without Augmentation on CIFAR-10")
print("=" * 60)

# Use a subset for faster experimentation
N_TRAIN = 10000
x_tr = x_train[:N_TRAIN]
y_tr = y_train[:N_TRAIN]

def build_cifar_cnn(with_augmentation=False):
    """
    Build a CNN for CIFAR-10.
    If with_augmentation=True, prepend Keras augmentation layers to the model.
    These layers are active during training (training=True) and disabled at inference.
    """
    inputs = keras.Input(shape=(32, 32, 3))

    # Augmentation layers — only active when model.fit() calls with training=True
    if with_augmentation:
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.1)(x)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomContrast(0.1)(x)
    else:
        x = inputs

    # CNN backbone
    x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2,2))(x)

    x = layers.Conv2D(64, (3,3), activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

EPOCHS = 20
BATCH  = 128

print(f"Training WITHOUT augmentation ({N_TRAIN} samples, {EPOCHS} epochs)...")
model_no_aug = build_cifar_cnn(with_augmentation=False)
hist_no_aug  = model_no_aug.fit(
    x_tr, y_tr,
    epochs=EPOCHS, batch_size=BATCH,
    validation_data=(x_test, y_test),
    verbose=1
)

print(f"\nTraining WITH augmentation ({N_TRAIN} samples, {EPOCHS} epochs)...")
model_aug  = build_cifar_cnn(with_augmentation=True)
hist_aug   = model_aug.fit(
    x_tr, y_tr,
    epochs=EPOCHS, batch_size=BATCH,
    validation_data=(x_test, y_test),
    verbose=1
)

# Report final test accuracy
_, acc_no_aug = model_no_aug.evaluate(x_test, y_test, verbose=0)
_, acc_aug    = model_aug.evaluate(x_test, y_test, verbose=0)

print(f"\nFinal Test Accuracy WITHOUT augmentation: {acc_no_aug:.4f}")
print(f"Final Test Accuracy WITH augmentation:    {acc_aug:.4f}")
print(f"Improvement from augmentation:             {(acc_aug - acc_no_aug)*100:+.2f} percentage points")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f"Lesson 14: Augmentation Impact on CIFAR-10\n"
             f"(Training on only {N_TRAIN} samples)",
             fontsize=13, fontweight='bold')

axes[0].plot(hist_no_aug.history['accuracy'],     'b-',  linewidth=2, label='Train (No Aug)')
axes[0].plot(hist_no_aug.history['val_accuracy'], 'b--', linewidth=2, label='Val (No Aug)')
axes[0].plot(hist_aug.history['accuracy'],        'g-',  linewidth=2, label='Train (With Aug)')
axes[0].plot(hist_aug.history['val_accuracy'],    'g--', linewidth=2, label='Val (With Aug)')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("Training & Validation Accuracy")
axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[0].annotate(f"No Aug: {acc_no_aug:.3f}\nWith Aug: {acc_aug:.3f}",
                 xy=(EPOCHS - 1, min(acc_no_aug, acc_aug)),
                 xytext=(EPOCHS - 8, 0.3),
                 arrowprops=dict(arrowstyle='->', color='black'),
                 fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

axes[1].plot(hist_no_aug.history['loss'],     'b-',  linewidth=2, label='Train Loss (No Aug)')
axes[1].plot(hist_no_aug.history['val_loss'], 'b--', linewidth=2, label='Val Loss (No Aug)')
axes[1].plot(hist_aug.history['loss'],        'g-',  linewidth=2, label='Train Loss (With Aug)')
axes[1].plot(hist_aug.history['val_loss'],    'g--', linewidth=2, label='Val Loss (With Aug)')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title("Training & Validation Loss\n(Augmentation raises train loss — harder examples)")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_14_augmentation_comparison.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_14_augmentation_comparison.png")
print("\n  Note: With augmentation, TRAINING loss may be higher (harder batches),")
print("  but VALIDATION accuracy is typically better — the model generalizes more.")

print("\n" + "=" * 60)
print("Lesson 14 Complete!")
print("Generated files:")
print("  - lesson_14_augmentation_gallery.png")
print("  - lesson_14_keras_augmentation_layers.png")
print("  - lesson_14_mixup_examples.png")
print("  - lesson_14_augmentation_comparison.png")
print("=" * 60)
