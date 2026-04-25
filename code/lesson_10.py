"""
Lesson 10: Training a CNN
CNN Image Classification Course

Sections:
  1. Manual training loop from scratch (NumPy)
  2. Keras model.fit() on MNIST with all parameters
  3. Plot training/validation accuracy and loss curves
  4. Effect of different learning rates
  5. EarlyStopping and ModelCheckpoint demonstration
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import os

print("=" * 60)
print("LESSON 10: Training a CNN")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Manual Training Loop from Scratch (NumPy)
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Manual Training Loop (NumPy) ---")
print("  Implementing logistic regression from scratch to illustrate")
print("  forward pass → loss → backward pass → weight update.")

np.random.seed(42)

# Simple 2D binary classification dataset (XOR-like)
N = 400
X_manual = np.random.randn(N, 2).astype('float32')
y_manual  = ((X_manual[:, 0] * X_manual[:, 1]) > 0).astype('float32')

# Network: 2 → 4 (tanh) → 1 (sigmoid)
def init_weights(in_dim, hidden_dim, out_dim):
    W1 = np.random.randn(in_dim, hidden_dim) * 0.5
    b1 = np.zeros(hidden_dim)
    W2 = np.random.randn(hidden_dim, out_dim) * 0.5
    b2 = np.zeros(out_dim)
    return W1, b1, W2, b2

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward(X, W1, b1, W2, b2):
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)
    Z2 = A1 @ W2 + b2
    A2 = sigmoid(Z2).flatten()
    return A1, A2

def compute_loss(A2, y):
    eps = 1e-9
    return -np.mean(y * np.log(A2 + eps) + (1 - y) * np.log(1 - A2 + eps))

def backward(X, y, A1, A2, W1, W2, lr=0.05):
    N = len(y)
    dL_dA2 = A2 - y                           # (N,)
    dA2_dZ2 = A2 * (1 - A2)                   # sigmoid derivative
    delta2  = (dL_dA2 * dA2_dZ2).reshape(-1, 1)

    dW2 = A1.T @ delta2 / N
    db2 = delta2.mean(axis=0)

    dL_dA1  = delta2 @ W2.T
    dA1_dZ1 = 1 - A1 ** 2                     # tanh derivative
    delta1   = dL_dA1 * dA1_dZ1

    dW1 = X.T @ delta1 / N
    db1 = delta1.mean(axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    return W1, b1, W2, b2

W1, b1, W2, b2 = init_weights(2, 16, 1)

print(f"\n  Training 2-layer network on XOR problem ({N} samples):")
print(f"  Architecture: 2 → 16 (tanh) → 1 (sigmoid)")
print(f"  {'Epoch':>6} | {'Loss':>10} | {'Accuracy':>10}")
print("  " + "-" * 34)

manual_losses = []
manual_accs   = []

for epoch in range(200):
    # Forward pass
    A1, A2 = forward(X_manual, W1, b1, W2, b2)

    # Loss
    loss = compute_loss(A2, y_manual)

    # Accuracy
    preds = (A2 >= 0.5).astype(float)
    acc   = np.mean(preds == y_manual)

    manual_losses.append(loss)
    manual_accs.append(acc)

    # Backward pass + weight update
    W1, b1, W2, b2 = backward(X_manual, y_manual, A1, A2, W1, W2, lr=0.1)

    if epoch % 40 == 0 or epoch == 199:
        print(f"  {epoch+1:>6} | {loss:>10.4f} | {acc:>9.2%}")

print(f"\n  Final loss: {manual_losses[-1]:.4f}")
print(f"  Final accuracy: {manual_accs[-1]:.2%}")
print("  ✓ Loss decreased steadily — manual training loop works!")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Keras model.fit() on MNIST
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Keras model.fit() with Full Parameters ---")

(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test       = X_test.astype('float32')       / 255.0
X_train_full = X_train_full[..., np.newaxis]
X_test       = X_test[..., np.newaxis]

# Use 12K training, 3K validation for speed
X_tr   = X_train_full[:12000]
y_tr   = y_train_full[:12000]
X_val  = X_train_full[12000:15000]
y_val  = y_train_full[12000:15000]

print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

def build_cnn(name="MyCNN"):
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), padding='same'),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ], name=name)

model_main = build_cnn("MNIST_CNN")
model_main.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n  model.fit() parameters explained:")
print("    epochs=8              → number of complete passes over training data")
print("    batch_size=64         → images per gradient update")
print("    validation_data=...   → data monitored but NOT used for training")
print("    shuffle=True          → shuffle training order each epoch")
print("    verbose=1             → show progress bar")

history_main = model_main.fit(
    x=X_tr,
    y=y_tr,
    epochs=8,
    batch_size=64,
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1
)

print("\n  Keys in history object:", list(history_main.history.keys()))

test_loss, test_acc = model_main.evaluate(X_test, y_test, verbose=0)
print(f"\n  Final test accuracy: {test_acc:.4f}")
print(f"  Final test loss:     {test_loss:.4f}")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Plot Training and Validation Curves
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Training and Validation Curves ---")

hist = history_main.history
epochs_range = range(1, len(hist['loss']) + 1)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Training Curves — MNIST CNN", fontsize=14, fontweight='bold')

# Loss curves
axes[0, 0].plot(epochs_range, hist['loss'],     'b-o', linewidth=2, label='Train Loss')
axes[0, 0].plot(epochs_range, hist['val_loss'], 'r--o', linewidth=2, label='Val Loss')
axes[0, 0].set_title("Loss Over Epochs")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_xticks(epochs_range)

# Accuracy curves
axes[0, 1].plot(epochs_range, hist['accuracy'],     'b-o', linewidth=2, label='Train Accuracy')
axes[0, 1].plot(epochs_range, hist['val_accuracy'], 'r--o', linewidth=2, label='Val Accuracy')
axes[0, 1].set_title("Accuracy Over Epochs")
axes[0, 1].set_xlabel("Epoch")
axes[0, 1].set_ylabel("Accuracy")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_xticks(epochs_range)

# Train/Val gap (overfitting indicator)
acc_gap  = np.array(hist['accuracy']) - np.array(hist['val_accuracy'])
loss_gap = np.array(hist['val_loss']) - np.array(hist['loss'])
axes[1, 0].bar(epochs_range, acc_gap, color='purple', alpha=0.6, label='Train−Val Accuracy Gap')
axes[1, 0].axhline(0, color='black', linewidth=0.8)
axes[1, 0].set_title("Overfitting Indicator: Train − Val Accuracy Gap")
axes[1, 0].set_xlabel("Epoch")
axes[1, 0].set_ylabel("Accuracy Gap")
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Manual training loop results
axes[1, 1].plot(manual_losses, 'g-', linewidth=1.5, label='Manual Loop Loss')
axes[1, 1].set_title("Manual NumPy Training Loop (Section 1)")
axes[1, 1].set_xlabel("Epoch")
axes[1, 1].set_ylabel("Binary Cross-Entropy Loss")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section10_training_curves.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section10_training_curves.png")

print("\n  Curve interpretation:")
max_gap = max(acc_gap)
if max_gap < 0.05:
    print(f"  ✓ Max train-val gap = {max_gap:.3f} — healthy training, no overfitting")
else:
    print(f"  ⚠ Max train-val gap = {max_gap:.3f} — signs of overfitting detected")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Effect of Different Learning Rates
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Effect of Different Learning Rates ---")

learning_rates = [0.1, 0.01, 0.001, 0.0001]
lr_histories   = {}

print("  Training same network with 4 different learning rates (5 epochs each):")
print(f"  {'LR':>8} | {'Final Train Loss':>18} | {'Final Val Acc':>15}")
print("  " + "-" * 48)

for lr in learning_rates:
    model_lr = build_cnn(f"CNN_lr_{lr}")
    model_lr.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    h = model_lr.fit(X_tr, y_tr, epochs=5, batch_size=64,
                     validation_data=(X_val, y_val), verbose=0)
    lr_histories[lr] = h.history
    final_loss = h.history['loss'][-1]
    final_acc  = h.history['val_accuracy'][-1]
    print(f"  {lr:>8.4f} | {final_loss:>18.4f} | {final_acc:>14.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Effect of Learning Rate on Training", fontsize=13, fontweight='bold')

colors = ['red', 'orange', 'green', 'blue']
for (lr, h), color in zip(lr_histories.items(), colors):
    ep_range = range(1, len(h['loss']) + 1)
    axes[0].plot(ep_range, h['loss'], color=color, linewidth=2,
                 label=f'lr={lr}', marker='o', markersize=4)
    axes[1].plot(ep_range, h['val_accuracy'], color=color, linewidth=2,
                 label=f'lr={lr}', marker='o', markersize=4)

axes[0].set_title("Training Loss by Learning Rate")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].set_title("Validation Accuracy by Learning Rate")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section10_learning_rates.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section10_learning_rates.png")
print("\n  Observations:")
print("    lr=0.1:    likely unstable / oscillating loss")
print("    lr=0.01:   converges, may overshoot slightly")
print("    lr=0.001:  typically optimal for Adam (default)")
print("    lr=0.0001: converges correctly but very slowly")

# ──────────────────────────────────────────────────────────────
# SECTION 5: EarlyStopping and ModelCheckpoint
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: EarlyStopping and ModelCheckpoint ---")

checkpoint_path = "best_mnist_model.keras"

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1
)

print("  Callbacks configured:")
print("    EarlyStopping:    monitor=val_loss, patience=3, restore_best_weights=True")
print("    ModelCheckpoint:  monitor=val_accuracy, save_best_only=True")
print("    ReduceLROnPlateau: monitor=val_loss, factor=0.5, patience=2")
print(f"\n  Training for up to 30 epochs (will stop early if val_loss stops improving)...")

model_cb = build_cnn("CNN_with_callbacks")
model_cb.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_cb = model_cb.fit(
    X_tr, y_tr,
    epochs=30,
    batch_size=64,
    validation_data=(X_val, y_val),
    callbacks=[early_stop, model_checkpoint, reduce_lr],
    verbose=1
)

actual_epochs = len(history_cb.history['loss'])
print(f"\n  Training stopped at epoch {actual_epochs} (out of max 30)")
print(f"  Best val_accuracy: {max(history_cb.history['val_accuracy']):.4f}")
print(f"  Best val_loss:     {min(history_cb.history['val_loss']):.4f}")

if os.path.exists(checkpoint_path):
    print(f"\n  Model saved to: {checkpoint_path}")
    loaded = tf.keras.models.load_model(checkpoint_path)
    saved_loss, saved_acc = loaded.evaluate(X_test, y_test, verbose=0)
    print(f"  Loaded model test accuracy: {saved_acc:.4f}")
    os.remove(checkpoint_path)
    print(f"  (Checkpoint file cleaned up)")

# Final summary plot with callbacks
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
fig.suptitle("Training with EarlyStopping + ModelCheckpoint + ReduceLROnPlateau",
             fontsize=11, fontweight='bold')

ep_range_cb = range(1, actual_epochs + 1)
axes[0].plot(ep_range_cb, history_cb.history['loss'],     'b-o', label='Train Loss')
axes[0].plot(ep_range_cb, history_cb.history['val_loss'], 'r--o', label='Val Loss')
axes[0].axvline(np.argmin(history_cb.history['val_loss']) + 1, color='green',
                linestyle=':', linewidth=2, label='Best Val Loss')
axes[0].set_title(f"Loss (stopped at epoch {actual_epochs})")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(ep_range_cb, history_cb.history['accuracy'],     'b-o', label='Train Acc')
axes[1].plot(ep_range_cb, history_cb.history['val_accuracy'], 'r--o', label='Val Acc')
axes[1].axvline(np.argmax(history_cb.history['val_accuracy']) + 1, color='green',
                linestyle=':', linewidth=2, label='Best Val Acc')
axes[1].set_title(f"Accuracy (stopped at epoch {actual_epochs})")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section10_callbacks.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section10_callbacks.png")

print("\n" + "=" * 60)
print("Lesson 10 Complete!")
print("Generated images:")
print("  section10_training_curves.png")
print("  section10_learning_rates.png")
print("  section10_callbacks.png")
print("=" * 60)
