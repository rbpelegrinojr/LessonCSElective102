"""
Lesson 10: Training a CNN
==========================
This module demonstrates the full CNN training process on CIFAR-10:
the training loop mechanics, learning rate effects, Keras callbacks
(ModelCheckpoint, EarlyStopping, ReduceLROnPlateau), learning curve
analysis, batch size comparison, and saving/loading trained models.
"""

# === IMPORTS ===
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    LambdaCallback
)

print("TensorFlow version:", tf.__version__)
print("GPU available:", len(tf.config.list_physical_devices('GPU')) > 0)
print("=" * 60)

# CIFAR-10 class names
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# =============================================================================
# === SECTION 1: LOAD AND PREPARE CIFAR-10 DATA ===
# =============================================================================

print("\n[SECTION 1] Loading and Preparing CIFAR-10")
print("-" * 60)

(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test       = y_test.flatten()

# Create validation split from training data
VAL_SIZE = 5000
x_val   = x_train_full[-VAL_SIZE:].astype('float32') / 255.0
y_val   = y_train_full[-VAL_SIZE:]
x_train = x_train_full[:-VAL_SIZE].astype('float32') / 255.0
y_train = y_train_full[:-VAL_SIZE]
x_test  = x_test.astype('float32') / 255.0

print(f"  Training:   {x_train.shape}, {len(x_train):,} samples")
print(f"  Validation: {x_val.shape},   {len(x_val):,}  samples")
print(f"  Test:       {x_test.shape},  {len(x_test):,} samples")


def build_cnn(input_shape=(32, 32, 3), num_classes=10):
    """
    Build a standard CNN for CIFAR-10 with 3 convolutional blocks.
    Architecture: Conv(32)→Pool → Conv(64)→Pool → Conv(128)→Pool → Dense → Output
    Uses BatchNormalization for training stability.
    """
    model = keras.Sequential([
        # ----- Block 1: detect edges and gradients -----
        layers.Conv2D(32, (3,3), padding='same', input_shape=input_shape),
        layers.BatchNormalization(),     # normalize activations → more stable training
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),       # 32×32 → 16×16
        layers.Dropout(0.25),            # regularization: drop 25% of units

        # ----- Block 2: detect textures and shapes -----
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),       # 16×16 → 8×8
        layers.Dropout(0.25),

        # ----- Block 3: detect object parts -----
        layers.Conv2D(128, (3,3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D(2, 2),       # 8×8 → 4×4
        layers.Dropout(0.25),

        # ----- Classification head -----
        layers.Flatten(),                # 4×4×128 = 2048
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),             # higher dropout before output
        layers.Dense(num_classes, activation='softmax'),
    ], name='CIFAR10_CNN')

    return model


# Print model summary once
model_demo = build_cnn()
print("\n  CNN Architecture Summary:")
model_demo.summary()
del model_demo  # free memory; we'll build fresh models for each experiment


# =============================================================================
# === SECTION 2: TRAINING WITH CALLBACKS ===
# =============================================================================

print("\n[SECTION 2] Training with ModelCheckpoint, EarlyStopping, ReduceLROnPlateau")
print("-" * 60)

# Ensure the code directory exists for saving checkpoints
os.makedirs('code', exist_ok=True)

CHECKPOINT_PATH = 'code/lesson_10_best_model.keras'
EPOCHS_MAX = 40    # maximum epochs; EarlyStopping will likely stop before this
BATCH_SIZE = 64
LEARNING_RATE = 0.001

# Define callbacks
callbacks = [
    # Save model only when validation loss improves
    ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        monitor='val_loss',         # what metric to watch
        save_best_only=True,        # don't overwrite with a worse model
        save_weights_only=False,    # save full model (architecture + weights)
        verbose=1,
        mode='min'                  # lower val_loss = better
    ),

    # Stop training early if val_loss doesn't improve for 8 epochs
    EarlyStopping(
        monitor='val_loss',
        patience=8,                 # wait 8 epochs before stopping
        restore_best_weights=True,  # revert weights to the best epoch
        verbose=1,
        mode='min'
    ),

    # Reduce learning rate when val_loss plateaus
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,         # multiply LR by 0.5 when triggered
        patience=4,         # wait 4 epochs of no improvement
        min_lr=1e-6,        # never go below this learning rate
        verbose=1,
        mode='min'
    ),

    # Custom callback to print LR at each epoch
    LambdaCallback(
        on_epoch_end=lambda epoch, logs: print(
            f"    Epoch {epoch+1:02d} → LR: "
            f"{float(tf.keras.backend.get_value(main_model.optimizer.learning_rate)):.6f}"
        )
    )
]

# Build and compile main model
main_model = build_cnn()
main_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Starting training (max {EPOCHS_MAX} epochs, early stopping enabled)...")
print(f"  Batch size: {BATCH_SIZE}, Initial LR: {LEARNING_RATE}")

history = main_model.fit(
    x_train, y_train,
    epochs=EPOCHS_MAX,
    batch_size=BATCH_SIZE,
    validation_data=(x_val, y_val),
    callbacks=callbacks,
    verbose=0   # suppress per-batch output; callback prints LR
)

actual_epochs = len(history.history['loss'])
print(f"\n  Training stopped after {actual_epochs} epochs (early stopping)")

# Evaluate final model (with restored best weights)
val_loss, val_acc = main_model.evaluate(x_val, y_val, verbose=0)
test_loss, test_acc = main_model.evaluate(x_test, y_test, verbose=0)
print(f"\n  Best model performance:")
print(f"    Validation accuracy: {val_acc:.4f}")
print(f"    Test accuracy:       {test_acc:.4f}")


# =============================================================================
# === SECTION 3: LEARNING CURVE ANALYSIS AND VISUALIZATION ===
# =============================================================================

print("\n[SECTION 3] Learning Curve Analysis")
print("-" * 60)

epochs_range = range(1, actual_epochs + 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Lesson 10: CNN Training on CIFAR-10 — Complete Analysis',
             fontsize=14, fontweight='bold')

# --- Loss curves ---
ax_loss = axes[0][0]
ax_loss.plot(epochs_range, history.history['loss'],
             'steelblue', linewidth=2, label='Training Loss')
ax_loss.plot(epochs_range, history.history['val_loss'],
             'steelblue', linewidth=2, linestyle='--', label='Validation Loss')
ax_loss.set_title('Loss Over Epochs', fontsize=12)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Cross-Entropy Loss')
ax_loss.legend()
ax_loss.grid(True, alpha=0.3)

# Annotate minimum validation loss
min_val_loss_epoch = np.argmin(history.history['val_loss']) + 1
min_val_loss = min(history.history['val_loss'])
ax_loss.axvline(x=min_val_loss_epoch, color='red', linewidth=1, linestyle=':',
                label=f'Best epoch: {min_val_loss_epoch}')
ax_loss.annotate(f'Best: {min_val_loss:.4f}',
                  xy=(min_val_loss_epoch, min_val_loss),
                  xytext=(min_val_loss_epoch + 1, min_val_loss + 0.05),
                  fontsize=9, color='red',
                  arrowprops=dict(arrowstyle='->', color='red'))

# --- Accuracy curves ---
ax_acc = axes[0][1]
ax_acc.plot(epochs_range, history.history['accuracy'],
            'darkorange', linewidth=2, label='Training Accuracy')
ax_acc.plot(epochs_range, history.history['val_accuracy'],
            'darkorange', linewidth=2, linestyle='--', label='Validation Accuracy')
ax_acc.axhline(y=test_acc, color='green', linewidth=1.5,
               linestyle=':', label=f'Test: {test_acc:.3f}')
ax_acc.set_title('Accuracy Over Epochs', fontsize=12)
ax_acc.set_xlabel('Epoch')
ax_acc.set_ylabel('Accuracy')
ax_acc.legend()
ax_acc.grid(True, alpha=0.3)

# --- Overfitting gap (train acc - val acc) ---
ax_gap = axes[1][0]
train_acc = np.array(history.history['accuracy'])
val_acc_arr = np.array(history.history['val_accuracy'])
gap = train_acc - val_acc_arr  # positive gap = overfitting

ax_gap.fill_between(epochs_range, gap, 0,
                     where=(gap > 0), alpha=0.4, color='red', label='Overfitting gap')
ax_gap.fill_between(epochs_range, gap, 0,
                     where=(gap <= 0), alpha=0.4, color='green', label='Underfitting')
ax_gap.plot(epochs_range, gap, 'black', linewidth=1.5)
ax_gap.axhline(y=0, color='black', linewidth=1)
ax_gap.set_title('Train-Val Accuracy Gap\n(Overfitting Monitor)', fontsize=12)
ax_gap.set_xlabel('Epoch')
ax_gap.set_ylabel('Train Acc − Val Acc')
ax_gap.legend()
ax_gap.grid(True, alpha=0.3)

# --- Per-class test accuracy ---
ax_class = axes[1][1]
y_pred = np.argmax(main_model.predict(x_test, verbose=0), axis=1)

class_accuracies = []
for cls in range(10):
    mask = y_test == cls
    cls_acc = np.mean(y_pred[mask] == y_test[mask])
    class_accuracies.append(cls_acc)

colors_bar = plt.cm.RdYlGn([acc for acc in class_accuracies])  # red=low, green=high
bars = ax_class.bar(CLASS_NAMES, class_accuracies, color=colors_bar, edgecolor='black')
ax_class.axhline(y=np.mean(class_accuracies), color='blue', linewidth=2,
                  linestyle='--', label=f'Mean: {np.mean(class_accuracies):.3f}')
ax_class.set_title('Per-Class Test Accuracy', fontsize=12)
ax_class.set_xlabel('Class')
ax_class.set_ylabel('Accuracy')
ax_class.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
ax_class.legend()
ax_class.set_ylim(0, 1.05)
for bar, acc in zip(bars, class_accuracies):
    ax_class.text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 0.01, f'{acc:.2f}',
                   ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('code/lesson_10_training_analysis.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_10_training_analysis.png")
plt.close()


# =============================================================================
# === SECTION 4: SAVE/RELOAD MODEL AND BATCH SIZE COMPARISON ===
# =============================================================================

print("\n[SECTION 4] Save/Reload Model and Batch Size Comparison")
print("-" * 60)

# --- Model save and reload ---
print("  Saving trained model...")
main_model.save('code/lesson_10_final_model.keras')
print("  Model saved to code/lesson_10_final_model.keras")

print("  Reloading model from disk...")
reloaded_model = keras.models.load_model('code/lesson_10_final_model.keras')
print("  Model reloaded successfully!")

# Verify reloaded model produces identical predictions
preds_original = main_model.predict(x_test[:100], verbose=0)
preds_reloaded = reloaded_model.predict(x_test[:100], verbose=0)
max_diff = np.max(np.abs(preds_original - preds_reloaded))
print(f"  Max prediction difference (original vs reloaded): {max_diff:.2e}")
print(f"  → Predictions are {'identical' if max_diff < 1e-5 else 'different'} ✓")

# --- Batch size comparison ---
print("\n  Comparing different batch sizes (short training for demo)...")
print("  (This shows the effect of batch size on convergence speed)")

BATCH_SIZES = [16, 64, 256]
QUICK_EPOCHS = 10  # fewer epochs for the comparison demo
batch_histories = {}

for bs in BATCH_SIZES:
    print(f"\n  Training with batch_size={bs}...")
    model_bs = build_cnn()
    model_bs.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model_bs.fit(
        x_train, y_train,
        epochs=QUICK_EPOCHS,
        batch_size=bs,
        validation_data=(x_val, y_val),
        verbose=0
    )
    final_val_acc = hist.history['val_accuracy'][-1]
    steps_per_epoch = len(x_train) // bs
    print(f"    Steps/epoch: {steps_per_epoch}, Final val acc: {final_val_acc:.4f}")
    batch_histories[bs] = hist
    del model_bs

# Plot batch size comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Lesson 10: Effect of Batch Size on Training',
             fontsize=14, fontweight='bold')

colors_bs = {16: 'steelblue', 64: 'darkorange', 256: 'green'}
epochs_q = range(1, QUICK_EPOCHS + 1)

for bs, hist in batch_histories.items():
    color = colors_bs[bs]
    label = f'batch={bs}'
    ax1.plot(epochs_q, hist.history['val_loss'],
             color=color, linewidth=2, label=label)
    ax2.plot(epochs_q, hist.history['val_accuracy'],
             color=color, linewidth=2, label=label)

ax1.set_title('Validation Loss by Batch Size', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title('Validation Accuracy by Batch Size', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/lesson_10_batch_comparison.png', dpi=120, bbox_inches='tight')
print("\n  Saved: code/lesson_10_batch_comparison.png")
plt.close()

# Print summary table
print("\n  Batch Size Comparison Summary:")
print(f"  {'Batch Size':<12} {'Val Acc (ep.10)':<18} {'Steps/Epoch':<15}")
print(f"  {'-'*12} {'-'*18} {'-'*15}")
for bs, hist in batch_histories.items():
    final_acc = hist.history['val_accuracy'][-1]
    steps = len(x_train) // bs
    print(f"  {bs:<12} {final_acc:<18.4f} {steps:<15}")

print("\n  Interpretation:")
print("  - Small batches (16):  more gradient noise, may generalize better,")
print("    but slower wall-clock time per epoch (more steps)")
print("  - Large batches (256): smoother gradients, faster per epoch,")
print("    but may converge to sharper minima (worse generalization)")
print("  - Batch 64-128 is typically the sweet spot for CNNs")

print("\n" + "=" * 60)
print("Lesson 10 complete! Generated files:")
print("  - code/lesson_10_training_analysis.png")
print("  - code/lesson_10_batch_comparison.png")
print("  - code/lesson_10_best_model.keras")
print("  - code/lesson_10_final_model.keras")
print("Key takeaways:")
print("  1. Training loop: forward pass → loss → backprop → weight update")
print("  2. Learning rate controls step size; 0.001 is a safe default for Adam")
print("  3. EarlyStopping prevents overfitting and saves time")
print("  4. ModelCheckpoint saves the best model (not necessarily the last)")
print("  5. ReduceLROnPlateau often triggers a second improvement phase")
print("  6. Batch size affects gradient noise, convergence speed, and generalization")
