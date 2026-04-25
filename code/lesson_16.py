"""
Lesson 16: Transfer Learning
============================
This module demonstrates transfer learning using pretrained CNN models
(VGG16 and MobileNetV2) from the Keras Applications library. We show:
  - How to load a pretrained base and freeze its weights
  - How to attach a custom classification head
  - How to train with feature extraction on CIFAR-10
  - How to compare transfer learning accuracy against a from-scratch baseline
  - How to use bottleneck features to speed up training
"""

# === STANDARD LIBRARY IMPORTS ===
import os
import time

# === NUMERICAL / PLOTTING IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe in all environments)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# === TENSORFLOW / KERAS IMPORTS ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, VGG16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

# Reproducibility seed
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("LESSON 16: Transfer Learning")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")
print(f"GPUs available: {len(tf.config.list_physical_devices('GPU'))}")


# ===================================================================
# SECTION 1: DATA LOADING AND PREPROCESSING
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 1: Data Loading and Preprocessing")
print("=" * 60)

# Load CIFAR-10 — 60,000 32×32 colour images across 10 classes
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = len(CLASS_NAMES)

print(f"Training images shape: {x_train.shape}")   # (50000, 32, 32, 3)
print(f"Test images shape:     {x_test.shape}")    # (10000, 32, 32, 3)
print(f"Class names: {CLASS_NAMES}")

# Use a smaller subset for faster demonstration
# In a real project you would use the full 50,000 training images
TRAIN_SUBSET = 10000
VAL_SUBSET   = 2000
x_train_sub = x_train[:TRAIN_SUBSET]
y_train_sub = y_train[:TRAIN_SUBSET]
x_val       = x_train[TRAIN_SUBSET:TRAIN_SUBSET + VAL_SUBSET]
y_val       = y_train[TRAIN_SUBSET:TRAIN_SUBSET + VAL_SUBSET]

print(f"\nUsing {TRAIN_SUBSET} training samples and {VAL_SUBSET} validation samples "
      f"for demonstration speed.")

# Target image size expected by MobileNetV2 (minimum 32×32, recommended 96+)
# We resize 32×32 CIFAR images to 96×96 so the pretrained base can extract
# meaningful high-level features (very small inputs lose information in deep nets).
IMG_SIZE = 96

def preprocess_for_mobilenet(images, labels):
    """Resize and apply MobileNetV2 normalisation (scales pixels to [-1, 1])."""
    # Cast to float32 before resizing
    images = tf.cast(images, tf.float32)
    # Resize from 32×32 to 96×96
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    # Apply MobileNetV2-specific normalisation
    images = mobilenet_preprocess(images)
    # One-hot encode labels
    labels = tf.squeeze(labels, axis=-1)
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels

BATCH_SIZE = 64

# Build tf.data pipelines — efficient for large datasets
train_ds = (
    tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
    .map(preprocess_for_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(buffer_size=2000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)    # overlap preprocessing with model execution
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess_for_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_for_mobilenet, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

print(f"\nResizing CIFAR-10 images from 32×32 to {IMG_SIZE}×{IMG_SIZE} for MobileNetV2.")

# Visualise sample training images
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("CIFAR-10 Sample Images", fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i])                        # original uint8 image
    ax.set_title(CLASS_NAMES[y_train[i][0]], fontsize=9)
    ax.axis('off')
plt.tight_layout()
plt.savefig('lesson_16_cifar10_samples.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_16_cifar10_samples.png")


# ===================================================================
# SECTION 2: EXPLORING PRETRAINED ARCHITECTURES
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 2: Exploring Pretrained Architectures")
print("=" * 60)

print("\n--- VGG16 Architecture ---")
# include_top=False removes the original 1000-class classification head
# weights='imagenet' downloads pretrained ImageNet weights automatically
vgg_base = VGG16(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,          # strip the original fully-connected head
    weights='imagenet'          # use weights trained on ImageNet
)
vgg_total_params  = vgg_base.count_params()
print(f"VGG16 base parameters:       {vgg_total_params:,}")
print(f"VGG16 base layers:           {len(vgg_base.layers)}")

print("\n--- MobileNetV2 Architecture ---")
mobilenet_base = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
mn_total_params = mobilenet_base.count_params()
print(f"MobileNetV2 base parameters: {mn_total_params:,}")
print(f"MobileNetV2 base layers:     {len(mobilenet_base.layers)}")

print(f"\nParameter ratio VGG16 / MobileNetV2: {vgg_total_params / mn_total_params:.1f}×")
print("MobileNetV2 achieves comparable accuracy with far fewer parameters!")

# Show architecture comparison table
print("\nArchitecture Comparison:")
print(f"{'Model':<15} {'Params':>12} {'Layers':>8} {'Best Use':<30}")
print("-" * 67)
print(f"{'VGG16':<15} {vgg_total_params:>12,} {len(vgg_base.layers):>8} {'Simple baseline, interpretable':<30}")
print(f"{'MobileNetV2':<15} {mn_total_params:>12,} {len(mobilenet_base.layers):>8} {'Mobile/edge, fast training':<30}")


# ===================================================================
# SECTION 3: FEATURE EXTRACTION WITH MOBILENETV2
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 3: Feature Extraction with MobileNetV2")
print("=" * 60)

def build_feature_extraction_model(base_model, num_classes=10):
    """
    Freeze the pretrained base and attach a custom classification head.
    Only the new head layers will be trained.
    """
    # Freeze ALL layers in the base — no gradients will flow through them
    base_model.trainable = False

    # Count trainable vs non-trainable params
    trainable_params     = sum(
        np.prod(v.shape) for v in base_model.trainable_weights
    )
    non_trainable_params = sum(
        np.prod(v.shape) for v in base_model.non_trainable_weights
    )
    print(f"  Base trainable params:     {trainable_params:,}")
    print(f"  Base non-trainable params: {non_trainable_params:,}")

    # Build the full model using the Functional API
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # Pass inputs through the frozen base (training=False keeps BatchNorm
    # layers in inference mode even during our training phase — this is
    # critical for stable feature extraction)
    x = base_model(inputs, training=False)

    # Global Average Pooling collapses the spatial dimensions:
    # (batch, 3, 3, 1280) → (batch, 1280)
    x = layers.GlobalAveragePooling2D()(x)

    # Dense feature layer with ReLU for non-linear transformation
    x = layers.Dense(256, activation='relu')(x)

    # Dropout for regularisation — prevents head from overfitting
    x = layers.Dropout(0.5)(x)

    # Output layer: one neuron per class, softmax for probabilities
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name='transfer_learning_model')
    return model


print("\nBuilding feature extraction model with MobileNetV2 base...")
transfer_model = build_feature_extraction_model(mobilenet_base, NUM_CLASSES)

# Show total parameter counts to verify freezing worked
total       = transfer_model.count_params()
trainable   = sum(np.prod(v.shape) for v in transfer_model.trainable_weights)
frozen      = total - trainable
print(f"\nTotal parameters:       {total:,}")
print(f"Trainable parameters:   {trainable:,}  (head only)")
print(f"Frozen parameters:      {frozen:,}   (MobileNetV2 base)")

# Compile with Adam optimiser and categorical crossentropy
transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
callbacks = [
    # Stop early if val_accuracy stops improving after 5 epochs
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    ),
    # Reduce LR if val_loss plateaus
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
]

print("\nTraining feature extraction model (only head layers)...")
print("This should be fast — only 333K parameters are trainable.\n")

start_time = time.time()
history_transfer = transfer_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks,
    verbose=1
)
transfer_time = time.time() - start_time
print(f"\nFeature extraction training time: {transfer_time:.1f} seconds")

# Evaluate on test set
test_loss_tl, test_acc_tl = transfer_model.evaluate(test_ds, verbose=0)
print(f"Transfer Learning Test Accuracy: {test_acc_tl:.4f} ({test_acc_tl*100:.1f}%)")


# ===================================================================
# SECTION 4: FROM-SCRATCH BASELINE AND COMPARISON
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 4: From-Scratch Baseline and Comparison")
print("=" * 60)

def build_scratch_model(input_shape, num_classes):
    """
    Simple CNN trained from scratch — no pretrained weights.
    Four convolutional blocks followed by a classification head.
    """
    model = keras.Sequential([
        # Block 1: 2 conv layers + pooling
        layers.Conv2D(32, 3, padding='same', activation='relu',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Block 2: wider filters to capture more complex patterns
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Block 3: even wider
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        # Classification head
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ], name='scratch_cnn')
    return model


# Note: scratch model uses simple [0,1] normalisation, not MobileNet normalisation
def preprocess_simple(images, labels):
    """Scale pixels to [0, 1] and resize."""
    images = tf.cast(images, tf.float32) / 255.0
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    labels = tf.squeeze(labels, axis=-1)
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels

train_ds_scratch = (
    tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
    .map(preprocess_simple, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
val_ds_scratch = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess_simple, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)
test_ds_scratch = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_simple, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
)

scratch_model = build_scratch_model(
    input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=NUM_CLASSES
)
scratch_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

scratch_params = scratch_model.count_params()
print(f"From-scratch CNN parameters: {scratch_params:,}")

scratch_callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=5, restore_best_weights=True
    )
]

print("\nTraining from-scratch CNN baseline...\n")
start_time = time.time()
history_scratch = scratch_model.fit(
    train_ds_scratch,
    validation_data=val_ds_scratch,
    epochs=15,
    callbacks=scratch_callbacks,
    verbose=1
)
scratch_time = time.time() - start_time

test_loss_sc, test_acc_sc = scratch_model.evaluate(test_ds_scratch, verbose=0)
print(f"\nFrom-Scratch Test Accuracy: {test_acc_sc:.4f} ({test_acc_sc*100:.1f}%)")

# --- Summary Comparison ---
print("\n" + "=" * 55)
print("ACCURACY COMPARISON SUMMARY")
print("=" * 55)
print(f"{'Method':<30} {'Test Acc':>10} {'Train Time':>12}")
print("-" * 55)
print(f"{'Transfer Learning (MNetV2)':<30} {test_acc_tl*100:>9.1f}% "
      f"{transfer_time:>10.0f}s")
print(f"{'From Scratch (Custom CNN)':<30} {test_acc_sc*100:>9.1f}% "
      f"{scratch_time:>10.0f}s")
gain = (test_acc_tl - test_acc_sc) * 100
print(f"\nTransfer learning advantage: {gain:+.1f} percentage points")
print("Even with only 10,000 training images, pretrained features help significantly!")

# --- Learning Curve Plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 16: Transfer Learning vs From Scratch", fontsize=14, fontweight='bold')

# Accuracy curves
ax = axes[0]
ax.plot(history_transfer.history['accuracy'],    label='TL - Train', color='blue')
ax.plot(history_transfer.history['val_accuracy'], label='TL - Val',   color='blue', linestyle='--')
ax.plot(history_scratch.history['accuracy'],     label='Scratch - Train', color='red')
ax.plot(history_scratch.history['val_accuracy'], label='Scratch - Val',   color='red', linestyle='--')
ax.set_title('Accuracy Curves')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
ax.grid(alpha=0.3)
# Mark final test accuracies
ax.axhline(test_acc_tl, color='blue', linestyle=':', alpha=0.7,
           label=f'TL Test: {test_acc_tl:.3f}')
ax.axhline(test_acc_sc, color='red',  linestyle=':', alpha=0.7,
           label=f'Scratch Test: {test_acc_sc:.3f}')

# Loss curves
ax = axes[1]
ax.plot(history_transfer.history['loss'],     label='TL - Train', color='blue')
ax.plot(history_transfer.history['val_loss'], label='TL - Val',   color='blue', linestyle='--')
ax.plot(history_scratch.history['loss'],     label='Scratch - Train', color='red')
ax.plot(history_scratch.history['val_loss'], label='Scratch - Val',   color='red', linestyle='--')
ax.set_title('Loss Curves')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lesson_16_learning_curves.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson_16_learning_curves.png")

# --- Bar chart comparison ---
fig, ax = plt.subplots(figsize=(7, 5))
methods  = ['Transfer Learning\n(MobileNetV2)', 'From Scratch\n(Custom CNN)']
accs     = [test_acc_tl * 100, test_acc_sc * 100]
colours  = ['steelblue', 'tomato']
bars = ax.bar(methods, accs, color=colours, edgecolor='black', width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=12)
ax.set_ylim(0, 100)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Transfer Learning vs From Scratch\n(10,000 training images)', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('lesson_16_accuracy_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_16_accuracy_comparison.png")

# Save the transfer model for use in Lesson 17
transfer_model.save('lesson_16_feature_extraction_model.keras')
print("\nSaved model: lesson_16_feature_extraction_model.keras")
print("(This model will be loaded by lesson_17.py for fine-tuning)")

print("\n" + "=" * 60)
print("LESSON 16 COMPLETE")
print("Key takeaway: Pretrained ImageNet weights dramatically boost")
print("accuracy on small datasets compared to training from scratch.")
print("=" * 60)
