"""
Lesson 15: Batch Normalization and Dropout
===========================================
This module demonstrates the practical effects of Batch Normalization and Dropout
on training speed, stability, and generalization of CNNs.
Topics covered:
  - Building CNNs with and without BatchNorm
  - Measuring training stability and convergence speed
  - Demonstrating BatchNorm behavior in training vs. inference mode
  - Applying Dropout layers and measuring their regularization effect
  - Visualizing the interaction between BatchNorm and Dropout
  - Spatial Dropout for convolutional feature maps
"""

# === Standard library and framework imports ===
import numpy as np
import matplotlib.pyplot as plt
import time

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("Lesson 15: Batch Normalization and Dropout")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")


# ===========================================================
# === SECTION 1: BATCH NORMALIZATION — TRAINING SPEED AND STABILITY ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 1: Batch Normalization — Training Speed & Stability")
print("=" * 60)

# Load and preprocess CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test  = y_test.flatten()
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print(f"Dataset loaded: {x_train.shape[0]} train, {x_test.shape[0]} test")

def build_cnn_no_batchnorm(learning_rate=0.001):
    """
    CNN without BatchNormalization.
    Training can be slower and more sensitive to learning rate choices.
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name="CNN_no_BN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def build_cnn_with_batchnorm(learning_rate=0.001):
    """
    CNN with BatchNormalization after each Conv layer (before activation — original paper style).
    
    The order here is: Conv → BN → ReLU
    BN normalizes the pre-activation values, so ReLU gets stable, zero-centered inputs.
    The learnable γ and β parameters allow the network to scale and shift after normalization.
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(32,32,3)),
        layers.BatchNormalization(),              # Normalize across the batch dimension
        layers.Activation('relu'),               # Apply ReLU after normalization

        layers.Conv2D(32, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(128, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Flatten(),

        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),

        layers.Dense(10, activation='softmax'),
    ], name="CNN_with_BN")
    # Note: use_bias=False because BN's β parameter plays the role of the bias
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

EPOCHS_BN = 20
BATCH_SIZE = 128

print(f"Training CNN WITHOUT BatchNorm for {EPOCHS_BN} epochs...")
model_no_bn = build_cnn_no_batchnorm(learning_rate=0.001)
print(f"  Parameters: {model_no_bn.count_params():,}")
t0 = time.time()
hist_no_bn = model_no_bn.fit(
    x_train, y_train,
    epochs=EPOCHS_BN, batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)
time_no_bn = time.time() - t0

print(f"\nTraining CNN WITH BatchNorm for {EPOCHS_BN} epochs...")
model_bn = build_cnn_with_batchnorm(learning_rate=0.001)
print(f"  Parameters: {model_bn.count_params():,}")
t0 = time.time()
hist_bn = model_bn.fit(
    x_train, y_train,
    epochs=EPOCHS_BN, batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)
time_bn = time.time() - t0

acc_no_bn, _ = model_no_bn.evaluate(x_test, y_test, verbose=0)[::-1]
acc_bn, _    = model_bn.evaluate(x_test, y_test, verbose=0)[::-1]

# Correct evaluation
_, test_acc_no_bn = model_no_bn.evaluate(x_test, y_test, verbose=0)
_, test_acc_bn    = model_bn.evaluate(x_test, y_test, verbose=0)

print(f"\nResults after {EPOCHS_BN} epochs:")
print(f"  No BN  → Test Acc: {test_acc_no_bn:.4f}, Training time: {time_no_bn:.0f}s")
print(f"  With BN → Test Acc: {test_acc_bn:.4f}, Training time: {time_bn:.0f}s")

# Visualize training curves: BN vs no BN
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 15: Effect of Batch Normalization on Training", fontsize=14, fontweight='bold')

axes[0].plot(hist_no_bn.history['val_accuracy'], 'r-o', markersize=4, linewidth=2, label='No BatchNorm')
axes[0].plot(hist_bn.history['val_accuracy'],    'b-o', markersize=4, linewidth=2, label='With BatchNorm')
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Validation Accuracy: BN vs No BN\n(BatchNorm enables faster convergence)")
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_no_bn.history['val_loss'], 'r-o', markersize=4, linewidth=2, label='No BatchNorm')
axes[1].plot(hist_bn.history['val_loss'],    'b-o', markersize=4, linewidth=2, label='With BatchNorm')
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Validation Loss: BN vs No BN")
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_15_batchnorm_effect.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_15_batchnorm_effect.png")


# ===========================================================
# === SECTION 2: BATCHNORM TRAINING vs. INFERENCE MODE ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 2: BatchNorm Training Mode vs. Inference Mode")
print("=" * 60)

# Build a simple model with BatchNorm to demonstrate the mode difference
demo_model = keras.Sequential([
    layers.Dense(64, use_bias=False, input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(10, activation='softmax'),
], name="BN_mode_demo")

# Create random input data
np.random.seed(0)
x_demo = np.random.randn(100, 20).astype("float32")

# Get predictions in TRAINING mode (uses batch statistics)
preds_train_mode = demo_model(x_demo, training=True).numpy()

# Get predictions in INFERENCE mode (uses running statistics)
preds_infer_mode = demo_model(x_demo, training=False).numpy()

# Compute the mean absolute difference
mean_diff = np.mean(np.abs(preds_train_mode - preds_infer_mode))
print(f"Mean absolute difference between training mode and inference mode predictions:")
print(f"  (Before model has learned running statistics): {mean_diff:.6f}")
print()
print("Key insight: Before training, running_mean=0 and running_var=1 (BN defaults).")
print("Training mode uses actual batch statistics; inference uses the running averages.")
print("After training, the running statistics converge to match the training data distribution.")
print("This gap closes as training progresses — always use model.predict() for evaluation.")

# Demonstrate correctly using training=False during custom evaluation
print("\nCorrect evaluation (training=False):")
correct_preds = demo_model(x_demo, training=False)
print(f"  Prediction shape: {correct_preds.shape}")
print(f"  Sum of probabilities for first sample: {correct_preds[0].numpy().sum():.6f} (should be 1.0)")

# Visualize the distribution of activations with and without BN
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Lesson 15: Batch Normalization Effect on Activation Distribution",
             fontsize=13, fontweight='bold')

# Model without BN: create a model and get intermediate activations
model_no_bn_simple = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(20,)),
    layers.Dense(64, activation='relu'),
], name="no_bn_dense")

model_bn_simple = keras.Sequential([
    layers.Dense(64, use_bias=False, input_shape=(20,)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dense(64, use_bias=False),
    layers.BatchNormalization(),
    layers.Activation('relu'),
], name="with_bn_dense")

# Get intermediate layer outputs
act_no_bn_layer1 = keras.Model(
    inputs=model_no_bn_simple.input,
    outputs=model_no_bn_simple.layers[0].output
)(x_demo, training=False).numpy()

act_bn_after_bn_layer1 = keras.Model(
    inputs=model_bn_simple.input,
    outputs=model_bn_simple.layers[1].output   # Output of BatchNormalization layer
)(x_demo, training=True).numpy()

act_bn_after_relu_layer1 = keras.Model(
    inputs=model_bn_simple.input,
    outputs=model_bn_simple.layers[2].output   # Output of Activation(relu) after BN
)(x_demo, training=True).numpy()

# Subplot 1: Distribution without BN
axes[0].hist(act_no_bn_layer1.flatten(), bins=50, color='salmon', edgecolor='black', alpha=0.7)
axes[0].set_title(f"Without BN — Layer 1 Activations\nMean={act_no_bn_layer1.mean():.2f}, "
                  f"Std={act_no_bn_layer1.std():.2f}")
axes[0].set_xlabel("Activation Value"); axes[0].set_ylabel("Count")
axes[0].grid(True, alpha=0.3)

# Subplot 2: After BN normalization (before activation)
axes[1].hist(act_bn_after_bn_layer1.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
axes[1].set_title(f"With BN — After BatchNorm\nMean≈0, Std≈1 (normalized)\n"
                  f"Mean={act_bn_after_bn_layer1.mean():.2f}, Std={act_bn_after_bn_layer1.std():.2f}")
axes[1].set_xlabel("Activation Value"); axes[1].set_ylabel("Count")
axes[1].grid(True, alpha=0.3)

# Subplot 3: After BN + ReLU
axes[2].hist(act_bn_after_relu_layer1.flatten(), bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
axes[2].set_title(f"With BN — After BatchNorm + ReLU\n(Zeros clamped by ReLU)\n"
                  f"Mean={act_bn_after_relu_layer1.mean():.2f}, Std={act_bn_after_relu_layer1.std():.2f}")
axes[2].set_xlabel("Activation Value"); axes[2].set_ylabel("Count")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_15_batchnorm_distributions.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_15_batchnorm_distributions.png")


# ===========================================================
# === SECTION 3: DROPOUT RATES AND THEIR EFFECT ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 3: Dropout Rate Effects")
print("=" * 60)

# Use a subset for faster experimentation
N_TRAIN = 10000
x_tr = x_train[:N_TRAIN]
y_tr = y_train[:N_TRAIN]

DROPOUT_EPOCHS = 20

def build_cnn_with_dropout(dropout_rate=0.0):
    """
    CNN with configurable Dropout rate.
    dropout_rate=0.0 means no dropout (baseline).
    Spatial Dropout is used in convolutional blocks; standard Dropout in Dense layers.
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.BatchNormalization(),
        # SpatialDropout2D drops entire feature maps (channels) — better for CNN layers
        # than standard Dropout which drops individual pixels (less meaningful)
        layers.SpatialDropout2D(dropout_rate / 2) if dropout_rate > 0 else layers.Lambda(lambda x: x),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.SpatialDropout2D(dropout_rate / 2) if dropout_rate > 0 else layers.Lambda(lambda x: x),
        layers.MaxPooling2D((2,2)),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        # Standard Dropout in Dense layers — drops individual neurons
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

dropout_rates = [0.0, 0.2, 0.4, 0.6]
dropout_histories = {}

for rate in dropout_rates:
    print(f"  Training with dropout_rate={rate}...")
    model_d = build_cnn_with_dropout(dropout_rate=rate)
    hist_d  = model_d.fit(
        x_tr, y_tr,
        epochs=DROPOUT_EPOCHS, batch_size=BATCH_SIZE,
        validation_data=(x_test, y_test),
        verbose=0
    )
    dropout_histories[rate] = hist_d
    best_val = max(hist_d.history['val_accuracy'])
    print(f"    Best val accuracy: {best_val:.4f}")

# Plot dropout comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Lesson 15: Effect of Dropout Rate on Training and Generalization",
             fontsize=13, fontweight='bold')

colors_d = ['black', 'blue', 'green', 'red']
rate_labels = {0.0: 'No Dropout (rate=0.0)', 0.2: 'Dropout=0.2',
               0.4: 'Dropout=0.4', 0.6: 'Dropout=0.6'}

# Plot 1: All validation accuracy curves
ax = axes[0, 0]
for rate, color in zip(dropout_rates, colors_d):
    ax.plot(dropout_histories[rate].history['val_accuracy'],
            color=color, linewidth=2, label=rate_labels[rate])
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
ax.set_title("Validation Accuracy by Dropout Rate")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 2: All training accuracy curves
ax = axes[0, 1]
for rate, color in zip(dropout_rates, colors_d):
    ax.plot(dropout_histories[rate].history['accuracy'],
            color=color, linewidth=2, label=rate_labels[rate])
ax.set_xlabel("Epoch"); ax.set_ylabel("Training Accuracy")
ax.set_title("Training Accuracy by Dropout Rate\n(Higher dropout → lower train accuracy)")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Plot 3: Train vs. Val for rate=0.0 (overfit baseline)
ax = axes[1, 0]
h = dropout_histories[0.0]
ax.plot(h.history['accuracy'],     'b-', linewidth=2, label='Train (No Dropout)')
ax.plot(h.history['val_accuracy'], 'r-', linewidth=2, label='Val (No Dropout)')
ax.fill_between(range(DROPOUT_EPOCHS),
                h.history['accuracy'],
                h.history['val_accuracy'],
                alpha=0.2, color='red', label='Overfitting Gap')
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("No Dropout: Train vs. Val — Overfitting Gap")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 4: Train vs. Val for best dropout rate (typically 0.4)
best_rate = max(dropout_rates, key=lambda r: max(dropout_histories[r].history['val_accuracy']))
ax = axes[1, 1]
h_best = dropout_histories[best_rate]
ax.plot(h_best.history['accuracy'],     'b-', linewidth=2, label=f'Train (Dropout={best_rate})')
ax.plot(h_best.history['val_accuracy'], 'g-', linewidth=2, label=f'Val (Dropout={best_rate})')
ax.fill_between(range(DROPOUT_EPOCHS),
                h_best.history['accuracy'],
                h_best.history['val_accuracy'],
                alpha=0.15, color='blue', label='Remaining Gap')
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title(f"Best Dropout (rate={best_rate}): Reduced Overfitting Gap")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_15_dropout_rates.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_15_dropout_rates.png")


# ===========================================================
# === SECTION 4: COMBINED BATCHNORM + DROPOUT ARCHITECTURE ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 4: Best-Practice Architecture — BatchNorm + Dropout")
print("=" * 60)

def build_best_practice_cnn():
    """
    Best-practice CNN architecture for CIFAR-10:
    - BatchNorm in convolutional blocks (after Conv, before activation)
    - SpatialDropout2D in conv blocks (drops feature maps, not pixels)
    - Standard Dropout in Dense blocks
    - No BatchNorm in the output layer
    
    This design follows widely-used modern CNN conventions.
    """
    model = keras.Sequential([
        # Block 1
        layers.Conv2D(32, (3,3), padding='same', use_bias=False, input_shape=(32,32,3)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(32, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.SpatialDropout2D(0.1),        # Drop 10% of feature maps

        # Block 2
        layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Conv2D(64, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),
        layers.SpatialDropout2D(0.2),        # Slightly higher rate for deeper block

        # Block 3
        layers.Conv2D(128, (3,3), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2,2)),

        # Classifier head
        layers.Flatten(),
        layers.Dense(256, use_bias=False),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),                 # Stronger dropout for the large Dense layer

        layers.Dense(10, activation='softmax'),    # No BN or Dropout on output layer
    ], name="BestPractice_CNN")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Building best-practice CNN with BatchNorm + Dropout...")
model_best = build_best_practice_cnn()
model_best.summary()
print(f"\nTotal parameters: {model_best.count_params():,}")

print(f"\nTraining best-practice CNN for {EPOCHS_BN} epochs on full CIFAR-10...")
hist_best = model_best.fit(
    x_train, y_train,
    epochs=EPOCHS_BN, batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

_, final_test_acc = model_best.evaluate(x_test, y_test, verbose=0)
print(f"Final Test Accuracy (Best Practice CNN): {final_test_acc:.4f}")

# Compare all four approaches side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Lesson 15: Comparing CNN Architectures\n"
             "No BN | With BN | With Dropout | BN + Dropout",
             fontsize=13, fontweight='bold')

# Use histories from earlier sections for comparison
# No BN: hist_no_bn, With BN: hist_bn, Best: hist_best
for hist, name, color, ls in [
    (hist_no_bn, 'No BN, No Dropout', 'red',    '-'),
    (hist_bn,    'BatchNorm only',    'blue',   '-'),
    (dropout_histories[0.4], f'Dropout=0.4 (no full BN)', 'orange', '-'),
    (hist_best,  'BN + Dropout (best)', 'green', '-'),
]:
    axes[0].plot(hist.history['val_accuracy'], color=color, linestyle=ls, linewidth=2, label=name)
    axes[1].plot(hist.history['val_loss'],     color=color, linestyle=ls, linewidth=2, label=name)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Validation Accuracy: Architecture Comparison")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Validation Loss: Architecture Comparison")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_15_architecture_comparison.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_15_architecture_comparison.png")

# Final summary
print("\n" + "-" * 55)
print(f"{'Architecture':<25} {'Best Val Acc':>12} {'Test Acc':>10}")
print("-" * 55)
_, t_no_bn = model_no_bn.evaluate(x_test, y_test, verbose=0)
_, t_bn    = model_bn.evaluate(x_test, y_test, verbose=0)
_, t_best  = model_best.evaluate(x_test, y_test, verbose=0)
print(f"{'No BN, No Dropout':<25} {max(hist_no_bn.history['val_accuracy']):>12.4f} {t_no_bn:>10.4f}")
print(f"{'BatchNorm only':<25} {max(hist_bn.history['val_accuracy']):>12.4f} {t_bn:>10.4f}")
print(f"{'Dropout only (0.4)':<25} {max(dropout_histories[0.4].history['val_accuracy']):>12.4f} {'N/A':>10}")
print(f"{'BN + Dropout (best)':<25} {max(hist_best.history['val_accuracy']):>12.4f} {t_best:>10.4f}")
print("-" * 55)

print("\n" + "=" * 60)
print("Lesson 15 Complete!")
print("Generated files:")
print("  - lesson_15_batchnorm_effect.png")
print("  - lesson_15_batchnorm_distributions.png")
print("  - lesson_15_dropout_rates.png")
print("  - lesson_15_architecture_comparison.png")
print("=" * 60)
