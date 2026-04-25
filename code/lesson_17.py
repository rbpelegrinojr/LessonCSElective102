"""
Lesson 17: Fine-Tuning Pretrained Models
=========================================
This module demonstrates how to move beyond feature extraction by gradually
unfreezing pretrained layers and fine-tuning them with a reduced learning rate.
We cover:
  - Loading the feature-extraction checkpoint from Lesson 16
  - Gradual unfreezing strategy (top block, then more blocks)
  - Discriminative learning rates across layer groups
  - Catastrophic forgetting: how to detect and avoid it
  - Stage-by-stage accuracy comparison and visualisation
"""

# === STANDARD LIBRARY IMPORTS ===
import os
import time

# === NUMERICAL / PLOTTING IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# === TENSORFLOW / KERAS IMPORTS ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("LESSON 17: Fine-Tuning Pretrained Models")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")


# ===================================================================
# SECTION 1: REBUILD DATASET AND FEATURE-EXTRACTION MODEL
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 1: Rebuild Dataset and Feature-Extraction Baseline")
print("=" * 60)

# --- Data setup (same as Lesson 16) ---
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE    = 96
BATCH_SIZE  = 64

# Subsets for demonstration speed
TRAIN_N, VAL_N = 10000, 2000
x_tr  = x_train[:TRAIN_N];   y_tr  = y_train[:TRAIN_N]
x_val = x_train[TRAIN_N:TRAIN_N + VAL_N];  y_val = y_train[TRAIN_N:TRAIN_N + VAL_N]

def make_dataset(x, y, shuffle=False):
    """Build a tf.data.Dataset with MobileNetV2 preprocessing."""
    def preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
        images = mobilenet_preprocess(images)          # scale to [-1, 1]
        labels = tf.squeeze(labels, axis=-1)
        labels = tf.one_hot(labels, NUM_CLASSES)
        return images, labels

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(buffer_size=2000)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(x_tr,    y_tr,   shuffle=True)
val_ds   = make_dataset(x_val,   y_val,  shuffle=False)
test_ds  = make_dataset(x_test,  y_test, shuffle=False)

# --- Try to load the saved model from Lesson 16; rebuild if not found ---
SAVED_MODEL_PATH = 'lesson_16_feature_extraction_model.keras'

if os.path.exists(SAVED_MODEL_PATH):
    print(f"\nLoading feature-extraction model from '{SAVED_MODEL_PATH}'...")
    stage1_model = keras.models.load_model(SAVED_MODEL_PATH)
    print("Loaded successfully.")
else:
    print(f"\n'{SAVED_MODEL_PATH}' not found — rebuilding feature extraction model...")
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights='imagenet')
    base.trainable = False
    inputs  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x       = base(inputs, training=False)
    x       = layers.GlobalAveragePooling2D()(x)
    x       = layers.Dense(256, activation='relu')(x)
    x       = layers.Dropout(0.5)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    stage1_model = keras.Model(inputs, outputs)
    stage1_model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='categorical_crossentropy', metrics=['accuracy']
    )
    stage1_model.fit(train_ds, validation_data=val_ds, epochs=15,
                     callbacks=[keras.callbacks.EarlyStopping(
                         monitor='val_accuracy', patience=5, restore_best_weights=True)],
                     verbose=1)
    stage1_model.save(SAVED_MODEL_PATH)
    print("Saved feature-extraction model.")

# Record baseline accuracy
_, acc_stage1 = stage1_model.evaluate(test_ds, verbose=0)
print(f"\nStage 1 (Feature Extraction) Test Accuracy: {acc_stage1*100:.2f}%")


# ===================================================================
# SECTION 2: GRADUAL UNFREEZING — TOP BLOCK FINE-TUNING
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 2: Gradual Unfreezing — Fine-Tune Top Block")
print("=" * 60)

# Helper: identify the MobileNetV2 base inside our model
def get_mobilenet_base(model):
    """Return the MobileNetV2 sub-model embedded in our transfer model."""
    for layer in model.layers:
        if 'mobilenetv2' in layer.name:
            return layer
    raise ValueError("MobileNetV2 base not found in model layers.")

# Clone the stage-1 model weights into a new model for stage 2
# We copy weights so we don't destroy the stage-1 checkpoint
import copy

# Rebuild architecture identical to stage1
base_s2 = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                      include_top=False, weights=None)   # random init first
inputs_s2  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x_s2       = base_s2(inputs_s2, training=False)
x_s2       = layers.GlobalAveragePooling2D()(x_s2)
x_s2       = layers.Dense(256, activation='relu')(x_s2)
x_s2       = layers.Dropout(0.5)(x_s2)
outputs_s2 = layers.Dense(NUM_CLASSES, activation='softmax')(x_s2)
stage2_model = keras.Model(inputs_s2, outputs_s2)

# Transfer weights from the saved stage-1 model
stage2_model.set_weights(stage1_model.get_weights())
print("Copied stage-1 weights into stage-2 model.")

# Inspect the MobileNetV2 base layer names to choose which to unfreeze
mn_base = get_mobilenet_base(stage2_model)
print(f"\nMobileNetV2 base has {len(mn_base.layers)} layers.")
print("First 5 layers:", [l.name for l in mn_base.layers[:5]])
print("Last  5 layers:", [l.name for l in mn_base.layers[-5:]])

# Unfreeze only the last N layers of the MobileNetV2 base
# The last ~20 layers correspond to the top inverted-residual block
UNFREEZE_FROM = len(mn_base.layers) - 20   # freeze everything before this index
mn_base.trainable = True                   # must set to True before setting per-layer

for i, layer in enumerate(mn_base.layers):
    if i < UNFREEZE_FROM:
        layer.trainable = False    # keep frozen
    else:
        # BatchNorm layers should stay in inference mode during fine-tuning
        # to use stable statistics from ImageNet training
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

# Count trainable parameters after unfreezing
trainable_s2 = sum(np.prod(v.shape) for v in stage2_model.trainable_weights)
total_s2     = stage2_model.count_params()
print(f"\nAfter unfreezing top block:")
print(f"  Trainable:     {trainable_s2:,}")
print(f"  Non-trainable: {total_s2 - trainable_s2:,}")
print(f"  Frozen layers: {UNFREEZE_FROM} / {len(mn_base.layers)}")

# Fine-tune with a MUCH smaller learning rate than stage-1
# Large LR would overwrite ImageNet knowledge (catastrophic forgetting)
stage2_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),  # 10× lower than stage 1
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nFine-tuning stage 2 (top block unfrozen, LR=1e-4)...")
callbacks_s2 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1
    )
]

history_s2 = stage2_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_s2,
    verbose=1
)

_, acc_stage2 = stage2_model.evaluate(test_ds, verbose=0)
print(f"\nStage 2 (Top Block Fine-Tuned) Test Accuracy: {acc_stage2*100:.2f}%")
improvement_s2 = (acc_stage2 - acc_stage1) * 100
print(f"Improvement over stage 1: {improvement_s2:+.2f} percentage points")


# ===================================================================
# SECTION 3: DEEPER FINE-TUNING — UNFREEZE MORE BLOCKS
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 3: Deeper Fine-Tuning — Unfreeze More Blocks")
print("=" * 60)

# Build stage-3 model by copying stage-2 weights
base_s3 = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                      include_top=False, weights=None)
inputs_s3  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x_s3       = base_s3(inputs_s3, training=False)
x_s3       = layers.GlobalAveragePooling2D()(x_s3)
x_s3       = layers.Dense(256, activation='relu')(x_s3)
x_s3       = layers.Dropout(0.5)(x_s3)
outputs_s3 = layers.Dense(NUM_CLASSES, activation='softmax')(x_s3)
stage3_model = keras.Model(inputs_s3, outputs_s3)
stage3_model.set_weights(stage2_model.get_weights())
print("Copied stage-2 weights into stage-3 model.")

mn_base_s3 = get_mobilenet_base(stage3_model)
# Unfreeze more layers — keep only the very bottom layers frozen
# (first ~50 layers learn basic edges and textures — rarely need updating)
UNFREEZE_FROM_S3 = 50
mn_base_s3.trainable = True

for i, layer in enumerate(mn_base_s3.layers):
    if i < UNFREEZE_FROM_S3:
        layer.trainable = False
    else:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

trainable_s3 = sum(np.prod(v.shape) for v in stage3_model.trainable_weights)
print(f"After deeper unfreezing:")
print(f"  Trainable:     {trainable_s3:,}")
print(f"  Frozen layers: {UNFREEZE_FROM_S3} / {len(mn_base_s3.layers)}")

# Even lower learning rate — more layers are at risk of forgetting
stage3_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),  # 2× lower than stage 2
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nFine-tuning stage 3 (more blocks unfrozen, LR=5e-5)...")
callbacks_s3 = [
    keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=7, restore_best_weights=True
    )
]

history_s3 = stage3_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    callbacks=callbacks_s3,
    verbose=1
)

_, acc_stage3 = stage3_model.evaluate(test_ds, verbose=0)
print(f"\nStage 3 (Deeper Fine-Tune) Test Accuracy: {acc_stage3*100:.2f}%")
improvement_s3 = (acc_stage3 - acc_stage1) * 100
print(f"Total improvement over stage 1: {improvement_s3:+.2f} percentage points")


# ===================================================================
# SECTION 4: VISUALISE FINE-TUNING PROGRESSION
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 4: Visualise Fine-Tuning Progression")
print("=" * 60)

# --- Stage summary table ---
print("\nFine-Tuning Stage Summary:")
print(f"{'Stage':<30} {'Test Accuracy':>14} {'Trainable Params':>18}")
print("-" * 65)
print(f"{'Stage 1: Feature Extraction':<30} {acc_stage1*100:>13.2f}% {'~333K':>18}")
print(f"{'Stage 2: Top Block Fine-Tune':<30} {acc_stage2*100:>13.2f}% {trainable_s2/1e6:>16.2f}M")
print(f"{'Stage 3: Deeper Fine-Tune':<30} {acc_stage3*100:>13.2f}% {trainable_s3/1e6:>16.2f}M")

# --- Combined validation accuracy plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 17: Fine-Tuning Progression", fontsize=14, fontweight='bold')

# Validation accuracy across all stages
ax = axes[0]
for hist, label, colour in [
    (history_s2, 'Stage 2: Top Block', 'steelblue'),
    (history_s3, 'Stage 3: Deeper',    'darkorange'),
]:
    epochs = range(1, len(hist.history['val_accuracy']) + 1)
    ax.plot(epochs, hist.history['val_accuracy'],
            label=label, color=colour, linewidth=2)
    ax.plot(epochs, hist.history['accuracy'],
            color=colour, linestyle='--', alpha=0.5)

ax.axhline(acc_stage1, color='green', linestyle=':', linewidth=2,
           label=f'Stage 1 baseline: {acc_stage1*100:.1f}%')
ax.set_title('Validation Accuracy by Fine-Tune Stage')
ax.set_xlabel('Epoch within Stage')
ax.set_ylabel('Accuracy')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Bar chart: final test accuracy per stage
ax = axes[1]
stages  = ['Stage 1\nFeature Extr.', 'Stage 2\nTop Block', 'Stage 3\nDeeper']
accs    = [acc_stage1*100, acc_stage2*100, acc_stage3*100]
colours = ['#2ca02c', '#1f77b4', '#ff7f0e']
bars = ax.bar(stages, accs, color=colours, edgecolor='black', width=0.5)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
            f'{acc:.1f}%', ha='center', fontweight='bold', fontsize=11)
ax.set_ylim(max(0, min(accs) - 5), min(100, max(accs) + 5))
ax.set_ylabel('Test Accuracy (%)')
ax.set_title('Test Accuracy at Each Fine-Tuning Stage')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('lesson_17_finetuning_progression.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_17_finetuning_progression.png")

# --- Catastrophic forgetting illustration ---
print("\nDemonstrating catastrophic forgetting (high LR + all layers unfrozen)...")
# Build a "bad" model: unfreeze everything + use high LR
base_bad = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights=None)
inputs_bad  = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x_bad       = base_bad(inputs_bad, training=True)   # training=True (bad practice here)
x_bad       = layers.GlobalAveragePooling2D()(x_bad)
x_bad       = layers.Dense(256, activation='relu')(x_bad)
x_bad       = layers.Dropout(0.5)(x_bad)
outputs_bad = layers.Dense(NUM_CLASSES, activation='softmax')(x_bad)
bad_model   = keras.Model(inputs_bad, outputs_bad)
bad_model.set_weights(stage1_model.get_weights())

bad_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-2),   # way too high!
    loss='categorical_crossentropy', metrics=['accuracy']
)

print("Using LR=1e-2 with all layers unfrozen (intentionally bad)...")
history_bad = bad_model.fit(
    train_ds, validation_data=val_ds,
    epochs=5, verbose=1
)

_, acc_bad = bad_model.evaluate(test_ds, verbose=0)
print(f"Catastrophic forgetting model accuracy: {acc_bad*100:.2f}%")
print(f"Compare to carefully fine-tuned stage 2: {acc_stage2*100:.2f}%")

# Plot catastrophic forgetting vs careful fine-tuning
fig, ax = plt.subplots(figsize=(8, 5))
good_epochs = range(1, len(history_s2.history['val_accuracy']) + 1)
bad_epochs  = range(1, len(history_bad.history['val_accuracy']) + 1)
ax.plot(good_epochs, history_s2.history['val_accuracy'],
        label='Careful fine-tuning (LR=1e-4)', color='blue', linewidth=2)
ax.plot(bad_epochs,  history_bad.history['val_accuracy'],
        label='Catastrophic forgetting (LR=1e-2, all unfrozen)', color='red',
        linewidth=2, linestyle='--')
ax.axhline(acc_stage1, color='green', linestyle=':', linewidth=2,
           label=f'Stage 1 baseline: {acc_stage1*100:.1f}%')
ax.set_title("Catastrophic Forgetting vs Careful Fine-Tuning", fontsize=13)
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Accuracy")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lesson_17_catastrophic_forgetting.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_17_catastrophic_forgetting.png")

# Save the best fine-tuned model for use in later lessons
best_acc  = max(acc_stage1, acc_stage2, acc_stage3)
best_model = {acc_stage1: stage1_model,
              acc_stage2: stage2_model,
              acc_stage3: stage3_model}[best_acc]
best_model.save('lesson_17_finetuned_model.keras')
print(f"\nSaved best fine-tuned model (acc={best_acc*100:.2f}%): "
      f"lesson_17_finetuned_model.keras")

print("\n" + "=" * 60)
print("LESSON 17 COMPLETE")
print("Key takeaway: Gradual unfreezing + low learning rate yields")
print("meaningful accuracy improvements without catastrophic forgetting.")
print("=" * 60)
