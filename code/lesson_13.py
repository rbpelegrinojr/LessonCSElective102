"""
Lesson 13: Overfitting and Regularization Techniques
======================================================
This module demonstrates overfitting, its diagnosis via learning curves,
and how regularization techniques help models generalize better.
Topics covered:
  - Deliberately inducing overfitting with a complex model on limited data
  - L1 and L2 regularization (with Keras regularizers)
  - Dropout regularization
  - Early stopping with best weight restoration
  - Comparing training vs. validation curves across regularization strategies
"""

# === Standard library and framework imports ===
import numpy as np
import matplotlib.pyplot as plt

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import EarlyStopping

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("Lesson 13: Overfitting and Regularization Techniques")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")


# ===========================================================
# === SECTION 1: DATA PREPARATION AND OVERFITTING DEMONSTRATION ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 1: Inducing Overfitting on Limited Data")
print("=" * 60)

# Load CIFAR-10
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test       = y_test.flatten()

# Normalize to [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test       = x_test.astype("float32") / 255.0

# Intentionally use a SMALL dataset to make overfitting easy to demonstrate
N_SMALL = 3000
x_small = x_train_full[:N_SMALL]
y_small = y_train_full[:N_SMALL]

# Use a validation split from the small set
VAL_SPLIT = 0.2
n_val = int(N_SMALL * VAL_SPLIT)
x_val, y_val = x_small[-n_val:], y_small[-n_val:]
x_tr,  y_tr  = x_small[:-n_val], y_small[:-n_val]

print(f"Small training set: {x_tr.shape[0]} samples (to induce overfitting)")
print(f"Validation set:     {x_val.shape[0]} samples")
print(f"Test set:           {x_test.shape[0]} samples")

EPOCHS = 40
BATCH  = 64

def build_overfit_model():
    """
    Deliberately oversized model (many parameters) with NO regularization.
    This will memorize the small training set, causing severe overfitting.
    """
    model = keras.Sequential([
        layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),    # Very large dense layer — no regularization
        layers.Dense(512,  activation='relu'),
        layers.Dense(10,   activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print(f"\nTraining overfit model (no regularization) for {EPOCHS} epochs...")
model_overfit = build_overfit_model()
print(f"Model parameters: {model_overfit.count_params():,}")

hist_overfit = model_overfit.fit(
    x_tr, y_tr,
    epochs=EPOCHS,
    batch_size=BATCH,
    validation_data=(x_val, y_val),
    verbose=0
)

final_train_acc = hist_overfit.history['accuracy'][-1]
final_val_acc   = hist_overfit.history['val_accuracy'][-1]
overfit_gap     = final_train_acc - final_val_acc
print(f"  Final Training Accuracy:   {final_train_acc:.4f}")
print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
print(f"  Overfitting Gap:           {overfit_gap:.4f}  ← This is the problem!")


# ===========================================================
# === SECTION 2: L1 AND L2 REGULARIZATION ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 2: L1 and L2 Regularization")
print("=" * 60)

def build_l2_model(lambda_val=0.001):
    """
    Same architecture as the overfit model but with L2 regularization
    on all Dense and Conv layers. L2 penalizes large weights by adding
    lambda * sum(w^2) to the loss.
    """
    model = keras.Sequential([
        layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(32,32,3),
                      kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(lambda_val)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(lambda_val)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Dense(512,  activation='relu', kernel_regularizer=regularizers.l2(lambda_val)),
        layers.Dense(10,   activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_l1_model(lambda_val=0.0001):
    """
    Same architecture with L1 regularization.
    L1 penalizes weights by adding lambda * sum(|w|) to the loss.
    L1 encourages sparsity — many weights become exactly zero.
    """
    model = keras.Sequential([
        layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(32,32,3),
                      kernel_regularizer=regularizers.l1(lambda_val)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l1(lambda_val)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l1(lambda_val)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l1(lambda_val)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l1(lambda_val)),
        layers.Dense(512,  activation='relu', kernel_regularizer=regularizers.l1(lambda_val)),
        layers.Dense(10,   activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print(f"Training L2 regularized model (lambda=0.001) for {EPOCHS} epochs...")
model_l2 = build_l2_model(lambda_val=0.001)
hist_l2  = model_l2.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                         validation_data=(x_val, y_val), verbose=0)
print(f"  L2 → Train: {hist_l2.history['accuracy'][-1]:.4f}, Val: {hist_l2.history['val_accuracy'][-1]:.4f}")

print(f"Training L1 regularized model (lambda=0.0001) for {EPOCHS} epochs...")
model_l1 = build_l1_model(lambda_val=0.0001)
hist_l1  = model_l1.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                         validation_data=(x_val, y_val), verbose=0)
print(f"  L1 → Train: {hist_l1.history['accuracy'][-1]:.4f}, Val: {hist_l1.history['val_accuracy'][-1]:.4f}")


# ===========================================================
# === SECTION 3: DROPOUT REGULARIZATION ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 3: Dropout Regularization")
print("=" * 60)

def build_dropout_model(dropout_rate=0.4):
    """
    Same architecture with Dropout layers added after Dense layers.
    Dropout randomly zeroes neurons during training, preventing co-adaptation
    and acting as an ensemble of many smaller networks.
    """
    model = keras.Sequential([
        layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(32,32,3)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu'),
        layers.Dropout(dropout_rate),            # Drop 40% of units during training
        layers.Dense(512, activation='relu'),
        layers.Dropout(dropout_rate),            # Drop 40% again
        layers.Dense(10, activation='softmax'),  # Never apply dropout to the output layer
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_combined_model(dropout_rate=0.4, l2_lambda=0.001):
    """
    Best-practice model: L2 regularization on weights + Dropout on activations.
    Both mechanisms work together for better generalization.
    """
    model = keras.Sequential([
        layers.Conv2D(64,  (3,3), activation='relu', padding='same', input_shape=(32,32,3),
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.MaxPooling2D((2,2)),
        layers.Flatten(),
        layers.Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(512,  activation='relu', kernel_regularizer=regularizers.l2(l2_lambda)),
        layers.Dropout(dropout_rate),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

print(f"Training model with Dropout (rate=0.4) for {EPOCHS} epochs...")
model_dropout = build_dropout_model(dropout_rate=0.4)
hist_dropout  = model_dropout.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                                   validation_data=(x_val, y_val), verbose=0)
print(f"  Dropout → Train: {hist_dropout.history['accuracy'][-1]:.4f}, Val: {hist_dropout.history['val_accuracy'][-1]:.4f}")

print(f"Training model with L2 + Dropout for {EPOCHS} epochs...")
model_combined = build_combined_model(dropout_rate=0.4, l2_lambda=0.001)
hist_combined  = model_combined.fit(x_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                                     validation_data=(x_val, y_val), verbose=0)
print(f"  L2+Dropout → Train: {hist_combined.history['accuracy'][-1]:.4f}, Val: {hist_combined.history['val_accuracy'][-1]:.4f}")


# ===========================================================
# === SECTION 4: EARLY STOPPING AND VISUAL COMPARISON ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 4: Early Stopping + Learning Curve Comparison")
print("=" * 60)

# Early stopping: stop when val_loss doesn't improve for 5 epochs
# restore_best_weights: rolls back to the best model checkpoint automatically
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print(f"Training combined model with Early Stopping (patience=5)...")
model_es = build_combined_model(dropout_rate=0.4, l2_lambda=0.001)
hist_es  = model_es.fit(
    x_tr, y_tr,
    epochs=100,              # Large epoch budget; early stopping will terminate early
    batch_size=BATCH,
    validation_data=(x_val, y_val),
    callbacks=[early_stop],
    verbose=0
)
stopped_epoch = len(hist_es.history['val_loss'])
print(f"  Training stopped at epoch {stopped_epoch} (out of 100)")
print(f"  Best val accuracy: {max(hist_es.history['val_accuracy']):.4f}")

# --- Plot all comparisons ---
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Lesson 13: Overfitting vs. Regularization — Learning Curves", fontsize=14, fontweight='bold')

all_histories = {
    'No Reg (Overfit)': hist_overfit,
    'L2 Reg':           hist_l2,
    'L1 Reg':           hist_l1,
    'Dropout':          hist_dropout,
    'L2 + Dropout':     hist_combined,
}
colors     = ['red', 'blue', 'purple', 'green', 'orange']
linestyles = ['-', '-', '-', '-', '-']

# Plot 1: All validation accuracy curves together
ax = axes[0, 0]
for (name, h), c in zip(all_histories.items(), colors):
    ax.plot(h.history['val_accuracy'], color=c, label=name, linewidth=2)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation Accuracy")
ax.set_title("Validation Accuracy: All Strategies")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 2: Overfit model — large gap between train and val
ax = axes[0, 1]
ax.plot(hist_overfit.history['accuracy'],     'b-', linewidth=2, label='Train Accuracy')
ax.plot(hist_overfit.history['val_accuracy'], 'r-', linewidth=2, label='Val Accuracy')
ax.fill_between(
    range(len(hist_overfit.history['accuracy'])),
    hist_overfit.history['accuracy'],
    hist_overfit.history['val_accuracy'],
    alpha=0.2, color='red', label='Overfitting Gap'
)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("Overfitting: Train vs. Val (No Regularization)")
ax.legend(); ax.grid(True, alpha=0.3)

# Plot 3: L2 + Dropout vs No Reg — combined is best
ax = axes[1, 0]
ax.plot(hist_overfit.history['val_accuracy'],  'r-', linewidth=2, label='No Reg (Val)')
ax.plot(hist_overfit.history['accuracy'],      'r--', linewidth=1, label='No Reg (Train)', alpha=0.5)
ax.plot(hist_combined.history['val_accuracy'], 'g-', linewidth=2, label='L2+Dropout (Val)')
ax.plot(hist_combined.history['accuracy'],     'g--', linewidth=1, label='L2+Dropout (Train)', alpha=0.5)
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title("No Regularization vs. L2+Dropout")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

# Plot 4: Early stopping demonstration
ax = axes[1, 1]
ax.plot(hist_es.history['val_accuracy'], 'purple', linewidth=2, label='Val Accuracy (with Early Stop)')
ax.plot(hist_es.history['accuracy'],     'purple', linewidth=1, linestyle='--', label='Train Accuracy', alpha=0.7)
ax.axvline(x=stopped_epoch - 1, color='red', linestyle='--', linewidth=2, label=f'Stopped at epoch {stopped_epoch}')
ax.axhline(y=max(hist_es.history['val_accuracy']), color='green', linestyle=':', linewidth=1.5,
           label=f"Best val acc = {max(hist_es.history['val_accuracy']):.4f}")
ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy")
ax.set_title(f"Early Stopping (patience=5, stopped at epoch {stopped_epoch})")
ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_13_regularization_comparison.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_13_regularization_comparison.png")

# --- Final summary table ---
print("\n" + "-" * 55)
print(f"{'Strategy':<22} {'Train Acc':>10} {'Val Acc':>10} {'Gap':>10}")
print("-" * 55)
for (name, h) in all_histories.items():
    tr  = h.history['accuracy'][-1]
    val = h.history['val_accuracy'][-1]
    gap = tr - val
    print(f"{name:<22} {tr:>10.4f} {val:>10.4f} {gap:>10.4f}")
# Early stopping uses best weights, so we use max val accuracy
tr_es  = hist_es.history['accuracy'][stopped_epoch-1]
val_es = max(hist_es.history['val_accuracy'])
print(f"{'Early Stop (best)':22} {tr_es:>10.4f} {val_es:>10.4f} {tr_es - val_es:>10.4f}")
print("-" * 55)

# Visualize bias-variance tradeoff conceptually
lambdas = np.logspace(-4, 0, 100)             # Range of regularization strengths
# Simulated curves for illustration (not from actual training)
bias_squared = 0.3 * (lambdas ** 0.5)        # Bias increases with over-regularization
variance     = 0.3 * np.exp(-3 * lambdas)    # Variance decreases with regularization
total_error  = bias_squared + variance + 0.05  # Total error has a U-shape minimum

fig, ax = plt.subplots(figsize=(9, 5))
ax.semilogx(lambdas, bias_squared, 'b--', linewidth=2, label='Bias² (underfitting)')
ax.semilogx(lambdas, variance,     'r--', linewidth=2, label='Variance (overfitting)')
ax.semilogx(lambdas, total_error,  'k-',  linewidth=3, label='Total Error')
optimal_idx = np.argmin(total_error)
ax.axvline(x=lambdas[optimal_idx], color='green', linestyle=':', linewidth=2,
           label=f'Optimal λ ≈ {lambdas[optimal_idx]:.4f}')
ax.set_xlabel("Regularization Strength λ (log scale)")
ax.set_ylabel("Error")
ax.set_title("Lesson 13: Bias-Variance Tradeoff vs. Regularization Strength")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.annotate('Underfitting\n(too much reg)', xy=(0.5, 0.4), fontsize=9, color='blue',
            ha='center')
ax.annotate('Overfitting\n(too little reg)', xy=(0.0001, 0.3), fontsize=9, color='red',
            ha='center')

plt.tight_layout()
plt.savefig("lesson_13_bias_variance_tradeoff.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_13_bias_variance_tradeoff.png")

print("\n" + "=" * 60)
print("Lesson 13 Complete!")
print("Generated files:")
print("  - lesson_13_regularization_comparison.png")
print("  - lesson_13_bias_variance_tradeoff.png")
print("=" * 60)
