"""
Lesson 11: Loss Functions & Optimizers
=======================================
Demonstrates cross-entropy loss, gradient descent from scratch,
optimizer comparison (SGD / Adam / RMSProp), 2-D loss landscape
visualization, and learning-rate schedules.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("=" * 60)
print("LESSON 11: Loss Functions & Optimizers")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Cross-Entropy Loss From Scratch vs Keras
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Cross-Entropy Loss From Scratch ---")

def cross_entropy_scratch(y_true_onehot, y_pred_probs, eps=1e-12):
    """Compute categorical cross-entropy loss manually.
    
    H(y, ŷ) = -Σ y_i * log(ŷ_i)
    
    Args:
        y_true_onehot: one-hot encoded ground truth, shape (N, C)
        y_pred_probs: predicted probabilities (softmax output), shape (N, C)
        eps: small constant to avoid log(0)
    Returns:
        mean loss over the batch
    """
    # Clip predictions to avoid log(0)
    y_pred_probs = np.clip(y_pred_probs, eps, 1.0 - eps)
    # Element-wise multiply ground truth and log predictions, then sum over classes
    per_example_loss = -np.sum(y_true_onehot * np.log(y_pred_probs), axis=1)
    return np.mean(per_example_loss)

def binary_cross_entropy_scratch(y_true, y_pred, eps=1e-12):
    """Binary cross-entropy: -(y*log(ŷ) + (1-y)*log(1-ŷ))"""
    y_pred = np.clip(y_pred, eps, 1.0 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

# --- Example 1: Categorical cross-entropy ---
# 3 examples, 4 classes
y_true_oh = np.array([[1, 0, 0, 0],   # true class = 0
                       [0, 1, 0, 0],   # true class = 1
                       [0, 0, 0, 1]])  # true class = 3

y_pred_good = np.array([[0.85, 0.05, 0.05, 0.05],  # confident and correct
                         [0.10, 0.75, 0.10, 0.05],
                         [0.05, 0.05, 0.05, 0.85]])

y_pred_bad  = np.array([[0.10, 0.30, 0.40, 0.20],   # uncertain and often wrong
                         [0.25, 0.25, 0.25, 0.25],
                         [0.40, 0.20, 0.30, 0.10]])

loss_good_scratch = cross_entropy_scratch(y_true_oh, y_pred_good)
loss_bad_scratch  = cross_entropy_scratch(y_true_oh, y_pred_bad)

# Compare against Keras's built-in categorical cross-entropy
keras_cce = keras.losses.CategoricalCrossentropy()
loss_good_keras = keras_cce(y_true_oh, y_pred_good).numpy()
loss_bad_keras  = keras_cce(y_true_oh, y_pred_bad).numpy()

print(f"Good predictions — Scratch: {loss_good_scratch:.4f} | Keras: {loss_good_keras:.4f}")
print(f"Bad  predictions — Scratch: {loss_bad_scratch:.4f} | Keras: {loss_bad_keras:.4f}")
assert abs(loss_good_scratch - loss_good_keras) < 1e-5, "Scratch and Keras results should match"
print("✓ Scratch implementation matches Keras")

# --- Binary cross-entropy ---
y_true_binary = np.array([1, 0, 1, 1, 0])
y_pred_binary = np.array([0.9, 0.2, 0.8, 0.6, 0.3])
bce_scratch = binary_cross_entropy_scratch(y_true_binary, y_pred_binary)
keras_bce   = keras.losses.BinaryCrossentropy()
bce_keras   = keras_bce(y_true_binary, y_pred_binary).numpy()
print(f"\nBinary CE — Scratch: {bce_scratch:.4f} | Keras: {bce_keras:.4f}")

# --- Sparse categorical cross-entropy (integer labels) ---
y_true_int  = np.array([0, 1, 3])  # integer class indices
scce = keras.losses.SparseCategoricalCrossentropy()
loss_scce_good = scce(y_true_int, y_pred_good).numpy()
print(f"Sparse Cat CE (same as Cat CE): {loss_scce_good:.4f}")
print("  → Sparse CE uses integer labels; Categorical CE uses one-hot vectors")

# --- MSE for contrast ---
mse_loss = keras.losses.MeanSquaredError()
# Using the same predictions as regression outputs (just for demo)
mse_val = mse_loss(y_true_oh.astype(float), y_pred_good).numpy()
print(f"\nMSE on same data (would be used for regression): {mse_val:.4f}")
print("  → MSE is NOT recommended for classification (use cross-entropy instead)")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Gradient Descent From Scratch on a Simple Parabola
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Gradient Descent From Scratch ---")

def parabola(w):
    """Objective function: f(w) = (w - 3)^2 + 2  (minimum at w=3, f=2)"""
    return (w - 3.0) ** 2 + 2.0

def parabola_gradient(w):
    """Analytical gradient: f'(w) = 2*(w - 3)"""
    return 2.0 * (w - 3.0)

def gradient_descent(start_w, learning_rate, n_steps):
    """Run gradient descent and track the trajectory."""
    w = start_w
    history = [w]
    for step in range(n_steps):
        grad = parabola_gradient(w)
        w = w - learning_rate * grad   # parameter update rule
        history.append(w)
        if step < 5 or step == n_steps - 1:
            print(f"  Step {step+1:3d}: w = {w:7.4f}, f(w) = {parabola(w):.4f}, grad = {grad:.4f}")
    return np.array(history)

print("Learning rate = 0.1:")
traj_01 = gradient_descent(start_w=10.0, learning_rate=0.1, n_steps=20)
print(f"  Final w = {traj_01[-1]:.4f} (true minimum = 3.0)")

print("\nLearning rate = 0.9 (large → oscillates):")
traj_09 = gradient_descent(start_w=10.0, learning_rate=0.9, n_steps=10)

# Plot gradient descent trajectories
w_range = np.linspace(-1, 12, 300)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: loss landscape with descent paths
axes[0].plot(w_range, parabola(w_range), 'k-', linewidth=2, label='f(w) = (w-3)² + 2')
axes[0].plot(traj_01, parabola(traj_01), 'bo-', markersize=6, label='LR=0.1 (smooth)')
axes[0].plot(traj_09, parabola(traj_09), 'rs--', markersize=6, label='LR=0.9 (oscillates)')
axes[0].axvline(3, color='green', linestyle=':', label='Minimum at w=3')
axes[0].set_xlabel('Weight (w)')
axes[0].set_ylabel('Loss f(w)')
axes[0].set_title('Gradient Descent on Parabola')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Right: w value over steps
steps = np.arange(len(traj_01))
axes[1].plot(steps, traj_01, 'bo-', markersize=5, label='LR=0.1')
axes[1].plot(np.arange(len(traj_09)), traj_09, 'rs--', markersize=5, label='LR=0.9')
axes[1].axhline(3.0, color='green', linestyle=':', label='True minimum')
axes[1].set_xlabel('Step')
axes[1].set_ylabel('w value')
axes[1].set_title('Weight Convergence Over Steps')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson11_gradient_descent.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson11_gradient_descent.png")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Compare SGD, Adam, RMSProp on MNIST
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Optimizer Comparison on MNIST ---")

# Load and preprocess MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
X_train = X_train.reshape(-1, 784)  # flatten
X_test  = X_test.reshape(-1, 784)

# Use a small subset for faster demo
N_TRAIN = 10000
X_tr, y_tr = X_train[:N_TRAIN], y_train[:N_TRAIN]
X_val, y_val = X_train[N_TRAIN:N_TRAIN+2000], y_train[N_TRAIN:N_TRAIN+2000]

def build_mlp():
    """Small MLP for MNIST digit classification."""
    return keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(784,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax'),
    ])

EPOCHS = 15
BATCH_SIZE = 64

optimizer_configs = {
    'SGD (lr=0.01)':      keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum':       keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'RMSProp (lr=0.001)': keras.optimizers.RMSprop(learning_rate=0.001),
    'Adam (lr=0.001)':    keras.optimizers.Adam(learning_rate=0.001),
}

histories = {}
for opt_name, optimizer in optimizer_configs.items():
    print(f"\n  Training with {opt_name}...")
    model = build_mlp()
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    hist = model.fit(X_tr, y_tr,
                     validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH_SIZE,
                     verbose=0)
    histories[opt_name] = hist
    final_val_acc = hist.history['val_accuracy'][-1]
    print(f"    Final val accuracy: {final_val_acc:.4f}")

# Plot loss curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'orange', 'green', 'red']

for (opt_name, hist), color in zip(histories.items(), colors):
    axes[0].plot(hist.history['loss'],     color=color, label=opt_name, linewidth=2)
    axes[1].plot(hist.history['val_accuracy'], color=color, label=opt_name, linewidth=2)

axes[0].set_title('Training Loss by Optimizer')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Accuracy by Optimizer')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson11_optimizer_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson11_optimizer_comparison.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: 2D Loss Landscape and Gradient Descent Path
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: 2D Loss Landscape Visualization ---")

def loss_2d(w1, w2):
    """2D bowl-shaped loss landscape with a slight ridge.
    Minimum near (2, 3).
    """
    return (w1 - 2.0)**2 + 2.0 * (w2 - 3.0)**2 + 0.3 * w1 * w2

def gradient_2d(w1, w2):
    """Analytical gradients of loss_2d."""
    dw1 = 2.0 * (w1 - 2.0) + 0.3 * w2
    dw2 = 4.0 * (w2 - 3.0) + 0.3 * w1
    return np.array([dw1, dw2])

def gd_2d(start, lr, n_steps, momentum=0.0):
    """Gradient descent in 2D with optional momentum."""
    w = np.array(start, dtype=float)
    velocity = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(n_steps):
        grad = gradient_2d(w[0], w[1])
        velocity = momentum * velocity - lr * grad  # momentum update
        w = w + velocity
        path.append(w.copy())
    return np.array(path)

# Generate the 2D loss surface
w1_vals = np.linspace(-2, 7, 200)
w2_vals = np.linspace(-1, 8, 200)
W1, W2 = np.meshgrid(w1_vals, w2_vals)
Z = loss_2d(W1, W2)

# Run GD from same starting point with different configs
path_vanilla  = gd_2d([-1, 0], lr=0.05, n_steps=40, momentum=0.0)
path_momentum = gd_2d([-1, 0], lr=0.05, n_steps=40, momentum=0.9)

print(f"  GD (no momentum) final: w1={path_vanilla[-1,0]:.3f}, w2={path_vanilla[-1,1]:.3f}")
print(f"  GD (momentum=0.9) final: w1={path_momentum[-1,0]:.3f}, w2={path_momentum[-1,1]:.3f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, path, title in zip(axes,
                            [path_vanilla, path_momentum],
                            ['Gradient Descent (No Momentum)', 'GD with Momentum=0.9']):
    contour = ax.contourf(W1, W2, Z, levels=30, cmap='viridis', alpha=0.7)
    ax.contour(W1, W2, Z, levels=15, colors='white', linewidths=0.5, alpha=0.4)
    plt.colorbar(contour, ax=ax, label='Loss')
    ax.plot(path[:, 0], path[:, 1], 'w-o', markersize=4, linewidth=2,
            label='GD path')
    ax.plot(path[0, 0], path[0, 1], 'r*', markersize=14, label='Start')
    ax.plot(2, 3, 'g*', markersize=14, label='True Minimum')
    ax.set_xlabel('w₁')
    ax.set_ylabel('w₂')
    ax.set_title(title)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig('lesson11_loss_landscape.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson11_loss_landscape.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Learning Rate Schedule Demonstration
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Learning Rate Schedules ---")

# Step Decay: halve the learning rate every 5 epochs
def step_decay(epoch, initial_lr=0.1, drop=0.5, epochs_drop=5):
    """Reduce learning rate by `drop` every `epochs_drop` epochs."""
    return initial_lr * (drop ** (epoch // epochs_drop))

# Cosine annealing schedule
def cosine_annealing(epoch, total_epochs=50, lr_min=1e-5, lr_max=0.1):
    """Cosine annealing from lr_max to lr_min over total_epochs."""
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / total_epochs))

# Warmup + cosine decay (popular with modern architectures)
def warmup_cosine(epoch, warmup_epochs=5, total_epochs=50, lr_max=0.01):
    """Linear warmup then cosine decay."""
    if epoch < warmup_epochs:
        return lr_max * (epoch + 1) / warmup_epochs
    adjusted = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return lr_max * 0.5 * (1 + np.cos(np.pi * adjusted))

epochs = np.arange(50)
lr_step   = [step_decay(e)        for e in epochs]
lr_cosine = [cosine_annealing(e)  for e in epochs]
lr_warmup = [warmup_cosine(e)     for e in epochs]

# Print first 10 epochs of each schedule
print("\n  Epoch | Step Decay | Cosine Ann. | Warmup+Cosine")
print("  " + "-" * 50)
for e in range(0, 50, 5):
    print(f"  {e:5d} | {lr_step[e]:.6f} | {lr_cosine[e]:.6f}  | {lr_warmup[e]:.6f}")

# Plot schedules
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(epochs, lr_step,   'b-o', markersize=4, label='Step Decay (×0.5 every 5 ep)')
ax.plot(epochs, lr_cosine, 'r-s', markersize=4, label='Cosine Annealing')
ax.plot(epochs, lr_warmup, 'g-^', markersize=4, label='Warmup + Cosine Decay')
ax.set_xlabel('Epoch')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedules')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_yscale('log')

plt.tight_layout()
plt.savefig('lesson11_lr_schedules.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson11_lr_schedules.png")

# --- Train a model with LR schedule using Keras callback ---
print("\n  Training model with ReduceLROnPlateau callback...")

model_sched = build_mlp()
model_sched.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ReduceLROnPlateau: halve LR when val_loss doesn't improve for 3 epochs
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6
)

hist_sched = model_sched.fit(
    X_tr, y_tr,
    validation_data=(X_val, y_val),
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    callbacks=[reduce_lr_cb], verbose=0
)

print(f"  Final val accuracy with LR schedule: {hist_sched.history['val_accuracy'][-1]:.4f}")

# Plot LR and accuracy over training
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(hist_sched.history['val_accuracy'], 'b-', linewidth=2)
axes[0].set_title('Validation Accuracy (ReduceLROnPlateau)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].grid(True, alpha=0.3)

# Learning rate history from callback
lr_history = hist_sched.history.get('lr', [0.01] * EPOCHS)
axes[1].plot(lr_history, 'r-o', linewidth=2, markersize=5)
axes[1].set_title('Learning Rate over Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Learning Rate')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson11_reduce_lr.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson11_reduce_lr.png")

print("\n" + "=" * 60)
print("Lesson 11 complete. Outputs saved:")
print("  lesson11_gradient_descent.png")
print("  lesson11_optimizer_comparison.png")
print("  lesson11_loss_landscape.png")
print("  lesson11_lr_schedules.png")
print("  lesson11_reduce_lr.png")
print("=" * 60)
