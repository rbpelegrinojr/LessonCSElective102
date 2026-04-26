"""
Lesson 16: Transfer Learning
Demonstrates loading pretrained models, feature extraction, and comparing
transfer learning vs training from scratch on CIFAR-10.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os

print("=" * 60)
print("Lesson 16: Transfer Learning")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# CIFAR-10 class names for reference
CIFAR10_CLASSES = ['airplane','automobile','bird','cat','deer',
                   'dog','frog','horse','ship','truck']

# ─────────────────────────────────────────────────────────────
# SECTION 1: Load MobileNetV2 and Explore Architecture
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 1: Loading MobileNetV2 Pretrained on ImageNet")
print("=" * 60)

# include_top=False removes the final ImageNet classification layer
# so we can attach our own custom head for a different number of classes
base_model = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,       # Remove the ImageNet 1000-class head
    weights='imagenet'       # Load weights trained on 1.2M images
)

print(f"MobileNetV2 total layers  : {len(base_model.layers)}")
print(f"MobileNetV2 output shape  : {base_model.output_shape}")

# Count trainable vs non-trainable parameters
total_params = base_model.count_params()
print(f"Total parameters          : {total_params:,}")

# Show the last 5 layers to understand what comes before our new head
print("\nLast 5 layers of MobileNetV2 backbone:")
for layer in base_model.layers[-5:]:
    print(f"  {layer.name:40s}  trainable={layer.trainable}")

# Compare multiple pretrained models available in Keras
print("\nComparison of pretrained models available in Keras:")
model_info = [
    ("MobileNetV2",  keras.applications.MobileNetV2),
    ("VGG16",        keras.applications.VGG16),
    ("ResNet50",     keras.applications.ResNet50),
]
print(f"  {'Model':<15} {'Parameters':>12}")
print("  " + "-" * 30)
for name, ModelClass in model_info:
    m = ModelClass(include_top=False, weights=None)
    print(f"  {name:<15} {m.count_params():>12,}")

# ─────────────────────────────────────────────────────────────
# SECTION 2: Feature Extraction — Freeze Base, Add Custom Head
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 2: Feature Extraction Setup")
print("=" * 60)

# Freeze ALL pretrained layers — their weights will not change during training
base_model.trainable = False
print(f"Base model trainable      : {base_model.trainable}")
print(f"Frozen layers             : {sum(1 for l in base_model.layers if not l.trainable)}")

# Build the full model: backbone + custom classification head
inputs = keras.Input(shape=(96, 96, 3))

# Always apply the model-specific preprocessing function!
# MobileNetV2 expects pixel values in [-1, 1], not [0, 255] or [0, 1]
x = keras.applications.mobilenet_v2.preprocess_input(inputs)

# Run through the frozen backbone; training=False ensures BatchNorm layers
# run in inference mode (use stored running statistics, not batch statistics)
x = base_model(x, training=False)

# GlobalAveragePooling2D: collapses spatial dims (7×7×1280 → 1280)
# More parameter-efficient than Flatten and less prone to overfitting
x = keras.layers.GlobalAveragePooling2D()(x)

# Regularization layer to prevent overfitting on small datasets
x = keras.layers.Dropout(0.2)(x)

# Output layer: 10 units for CIFAR-10's 10 classes
outputs = keras.layers.Dense(10, activation='softmax')(x)

transfer_model = keras.Model(inputs, outputs, name='TransferLearningModel')

transfer_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Only the new head is trainable — that's just 2 layers
trainable = sum(1 for l in transfer_model.layers if l.trainable)
print(f"Trainable layers in full model: {trainable}")

# Print a concise summary showing input/output shapes
print("\nModel summary (head only, backbone condensed):")
transfer_model.summary(line_length=70, expand_nested=False)

# ─────────────────────────────────────────────────────────────
# SECTION 3: Feature Extraction on CIFAR-10
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 3: Training Feature Extractor on CIFAR-10")
print("=" * 60)

# Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
print(f"CIFAR-10 training set : {x_train.shape}  labels: {y_train.shape}")
print(f"CIFAR-10 test set     : {x_test.shape}")

# Use a subset to keep training fast for demonstration
SUBSET = 5000
x_train_sub = x_train[:SUBSET].astype('float32')
y_train_sub = y_train[:SUBSET]
x_val_sub   = x_test[:1000].astype('float32')
y_val_sub   = y_test[:1000]
print(f"Using {SUBSET} training samples and 1000 validation samples")

# CIFAR-10 images are 32×32; MobileNetV2 needs at least 32×32 but works
# better with 96×96 due to its architecture — resize using tf.data pipeline
def resize_and_cast(image, label):
    """Resize images to 96×96 for MobileNetV2 compatibility."""
    image = tf.image.resize(image, [96, 96])
    return image, label

BATCH_SIZE = 32
AUTOTUNE   = tf.data.AUTOTUNE

# Build efficient tf.data pipelines with resize, cache, and prefetch
train_ds = (tf.data.Dataset.from_tensor_slices((x_train_sub, y_train_sub))
            .map(resize_and_cast, num_parallel_calls=AUTOTUNE)
            .shuffle(1000)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

val_ds = (tf.data.Dataset.from_tensor_slices((x_val_sub, y_val_sub))
          .map(resize_and_cast, num_parallel_calls=AUTOTUNE)
          .batch(BATCH_SIZE)
          .prefetch(AUTOTUNE))

print(f"\nTraining feature extractor head for 5 epochs...")
history_transfer = transfer_model.fit(
    train_ds,
    epochs=5,
    validation_data=val_ds,
    verbose=1
)

final_train_acc = history_transfer.history['accuracy'][-1]
final_val_acc   = history_transfer.history['val_accuracy'][-1]
print(f"\nFeature extraction — Final train acc : {final_train_acc:.4f}")
print(f"Feature extraction — Final val acc   : {final_val_acc:.4f}")

# ─────────────────────────────────────────────────────────────
# SECTION 4: Visualize Learned Feature Maps
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 4: Visualizing Pretrained Feature Maps")
print("=" * 60)

# Build an activation model that outputs intermediate layer activations
# Pick the first and a mid-level conv layer from MobileNetV2
target_layer_names = []
for layer in base_model.layers:
    if 'Conv' in layer.__class__.__name__:
        target_layer_names.append(layer.name)

# Use the first, a middle, and the last conv layer
selected = [target_layer_names[0],
            target_layer_names[len(target_layer_names)//2],
            target_layer_names[-1]]
print(f"Visualizing layers: {selected}")

# Create a model that outputs activations from selected layers
activation_outputs = [base_model.get_layer(name).output for name in selected]
activation_model = keras.Model(inputs=base_model.input, outputs=activation_outputs)

# Prepare one sample image (resize a CIFAR-10 test image)
sample_img = tf.image.resize(x_test[0:1].astype('float32'), [96, 96])
sample_img_processed = keras.applications.mobilenet_v2.preprocess_input(sample_img)

# Get activations for the sample image
activations = activation_model.predict(sample_img_processed, verbose=0)

fig, axes = plt.subplots(3, 8, figsize=(16, 7))
fig.suptitle('Feature Maps from Pretrained MobileNetV2\n'
             '(Early → Mid → Late Layers)', fontsize=13)

for row_idx, (layer_name, act) in enumerate(zip(selected, activations)):
    # act shape: (1, H, W, C) — pick first 8 channels
    num_channels = min(8, act.shape[-1])
    for col_idx in range(8):
        ax = axes[row_idx, col_idx]
        if col_idx < num_channels:
            # Normalize each feature map independently for visibility
            feature_map = act[0, :, :, col_idx]
            vmin, vmax = feature_map.min(), feature_map.max()
            if vmax > vmin:
                feature_map = (feature_map - vmin) / (vmax - vmin)
            ax.imshow(feature_map, cmap='viridis')
        ax.axis('off')
    axes[row_idx, 0].set_ylabel(
        layer_name.split('_')[0] + '\n(layer ' + str(row_idx+1) + ')',
        fontsize=8, rotation=90
    )

plt.tight_layout()
plt.savefig('section16_feature_maps.png', dpi=80, bbox_inches='tight')
plt.close()
print("Saved: section16_feature_maps.png")

# ─────────────────────────────────────────────────────────────
# SECTION 5: Transfer Learning vs Training from Scratch
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SECTION 5: Transfer Learning vs Training from Scratch")
print("=" * 60)

# Build a small CNN trained from scratch for comparison
scratch_model = keras.Sequential([
    keras.layers.Input(shape=(32, 32, 3)),
    keras.layers.Rescaling(1.0 / 255),
    keras.layers.Conv2D(32, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax'),
], name='ScratchCNN')

scratch_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"Scratch CNN parameters  : {scratch_model.count_params():,}")

# Use the same 1000 samples (very small dataset) for a fair comparison
SMALL_N = 1000
x_small = x_train[:SMALL_N].astype('float32')
y_small = y_train[:SMALL_N]

small_train_ds = (tf.data.Dataset.from_tensor_slices((x_small, y_small))
                  .shuffle(500).batch(32).prefetch(AUTOTUNE))

small_val_ds = (tf.data.Dataset.from_tensor_slices(
                    (x_test[:500].astype('float32'), y_test[:500]))
                .batch(32).prefetch(AUTOTUNE))

print(f"\nTraining from-scratch CNN on {SMALL_N} samples for 5 epochs...")
history_scratch = scratch_model.fit(
    small_train_ds, epochs=5, validation_data=small_val_ds, verbose=1
)

# Retrain a fresh transfer model on the same tiny dataset
transfer_model_small = keras.Sequential([
    keras.Input(shape=(96, 96, 3)),
    keras.layers.Lambda(keras.applications.mobilenet_v2.preprocess_input),
    keras.applications.MobileNetV2(include_top=False, weights='imagenet',
                                   input_shape=(96, 96, 3)),
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax'),
], name='TransferSmall')
transfer_model_small.layers[2].trainable = False  # Freeze MobileNetV2

transfer_model_small.compile(
    optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']
)

def resize_only(image, label):
    return tf.image.resize(image, [96, 96]), label

small_train_96 = (tf.data.Dataset.from_tensor_slices((x_small, y_small))
                  .map(resize_only).shuffle(500).batch(32).prefetch(AUTOTUNE))
small_val_96   = (tf.data.Dataset.from_tensor_slices(
                      (x_test[:500].astype('float32'), y_test[:500]))
                  .map(resize_only).batch(32).prefetch(AUTOTUNE))

print(f"\nTraining transfer model on {SMALL_N} samples for 5 epochs...")
history_transfer_small = transfer_model_small.fit(
    small_train_96, epochs=5, validation_data=small_val_96, verbose=1
)

# Plot side-by-side comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
epochs = range(1, 6)

ax1.plot(epochs, history_scratch.history['val_accuracy'],
         'r-o', label='Scratch CNN')
ax1.plot(epochs, history_transfer_small.history['val_accuracy'],
         'b-o', label='Transfer Learning')
ax1.set_title(f'Validation Accuracy\n(Only {SMALL_N} Training Samples)')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, history_scratch.history['val_loss'],
         'r-o', label='Scratch CNN')
ax2.plot(epochs, history_transfer_small.history['val_loss'],
         'b-o', label='Transfer Learning')
ax2.set_title(f'Validation Loss\n(Only {SMALL_N} Training Samples)')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('section16_transfer_vs_scratch.png', dpi=80, bbox_inches='tight')
plt.close()
print("Saved: section16_transfer_vs_scratch.png")

scratch_final   = history_scratch.history['val_accuracy'][-1]
transfer_final  = history_transfer_small.history['val_accuracy'][-1]
print(f"\nFinal val accuracy — Scratch     : {scratch_final:.4f}")
print(f"Final val accuracy — Transfer    : {transfer_final:.4f}")
print(f"Transfer learning advantage      : {transfer_final - scratch_final:+.4f}")
print("\nKey takeaway: Transfer learning reaches higher accuracy with less data")
print("because pretrained ImageNet features (edges, textures) apply broadly.")
