"""
Lesson 11: Loss Functions and Optimizers
=========================================
This module demonstrates the role of loss functions in neural network training and
compares different optimizers (SGD, Adam, RMSProp) on MNIST. It includes:
  - Manual gradient descent on a simple function
  - Comparison of loss functions (MSE vs Cross-Entropy)
  - Visual comparison of optimizer convergence
  - Learning rate effects on training stability
  - Learning rate scheduling strategies
"""

# === Standard library and framework imports ===
import numpy as np                               # Numerical computations
import matplotlib.pyplot as plt                  # Plotting and visualization
import matplotlib.gridspec as gridspec           # Complex subplot layouts

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, losses
from tensorflow.keras.datasets import mnist
from tensorflow.keras.callbacks import LearningRateScheduler, ReduceLROnPlateau

print("=" * 60)
print("Lesson 11: Loss Functions and Optimizers")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# Suppress TF info/warning logs for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ===========================================================
# === SECTION 1: MANUAL GRADIENT DESCENT ON A SIMPLE FUNCTION ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 1: Manual Gradient Descent Visualization")
print("=" * 60)

def f(x):
    """Quadratic function f(x) = (x - 3)^2 + 2 — minimum at x=3."""
    return (x - 3.0) ** 2 + 2.0

def df(x):
    """Analytical derivative of f: df/dx = 2*(x - 3)."""
    return 2.0 * (x - 3.0)

def gradient_descent(start_x, learning_rate, n_steps=50):
    """
    Run gradient descent from start_x for n_steps.
    Returns the path of x values and corresponding loss values.
    """
    x = start_x
    path_x = [x]
    path_loss = [f(x)]
    for _ in range(n_steps):
        gradient = df(x)                         # Compute gradient at current x
        x = x - learning_rate * gradient         # Update x: move opposite to gradient
        path_x.append(x)
        path_loss.append(f(x))
    return np.array(path_x), np.array(path_loss)

print("Running gradient descent with different learning rates on f(x) = (x-3)^2 + 2")

# Test three different learning rates to show their effects
learning_rates = {
    "Too Low (lr=0.01)":  0.01,
    "Just Right (lr=0.1)": 0.1,
    "Too High (lr=1.5)":  1.5,
}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 11: Gradient Descent — Effect of Learning Rate", fontsize=14, fontweight='bold')

# Left plot: the function landscape
x_vals = np.linspace(-3, 9, 300)
axes[0].plot(x_vals, f(x_vals), 'k-', linewidth=2, label='f(x) = (x-3)² + 2')
axes[0].axvline(x=3, color='green', linestyle='--', alpha=0.5, label='True minimum x=3')
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
axes[0].set_title("Loss Landscape and Gradient Descent Paths")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

colors = ['blue', 'orange', 'red']
for (name, lr), color in zip(learning_rates.items(), colors):
    path_x, path_loss = gradient_descent(start_x=8.0, learning_rate=lr, n_steps=30)
    # Plot the path on the function landscape
    axes[0].plot(path_x[:20], f(path_x[:20]), 'o-', color=color, label=name, alpha=0.7, markersize=4)
    # Plot the loss over steps
    axes[1].plot(path_loss[:30], 'o-', color=color, label=name, markersize=4)
    final_val = path_x[-1] if np.isfinite(path_x[-1]) else float('nan')
    print(f"  {name}: final x = {final_val:.4f}, final loss = {path_loss[-1]:.4f}")

axes[0].legend(loc='upper left', fontsize=8)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].set_title("Loss Value Over Gradient Descent Steps")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)
axes[1].set_ylim(-1, 50)   # Cap the y-axis so diverging lr=1.5 doesn't dominate

plt.tight_layout()
plt.savefig("lesson_11_gradient_descent.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_gradient_descent.png")


# ===========================================================
# === SECTION 2: LOSS FUNCTION COMPARISON ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 2: Loss Function Comparison (MSE vs Cross-Entropy)")
print("=" * 60)

# --- Load and preprocess MNIST ---
print("Loading MNIST dataset...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] and reshape to (N, 28, 28, 1)
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test  = np.expand_dims(x_test, -1)

# One-hot encode labels for cross-entropy comparison
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_test_oh  = keras.utils.to_categorical(y_test, 10)

# Use a small subset for quick experiments
N_TRAIN = 10000
N_TEST  = 2000
x_tr = x_train[:N_TRAIN]
x_te = x_test[:N_TEST]
y_tr_sparse = y_train[:N_TRAIN]           # Integer labels for sparse loss
y_tr_oh     = y_train_oh[:N_TRAIN]        # One-hot labels for MSE comparison
y_te_oh     = y_test_oh[:N_TEST]

def build_simple_cnn(loss_fn, optimizer_name='adam', learning_rate=0.001):
    """
    Build a small CNN for MNIST.
    loss_fn: a Keras loss function name or object
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),   # Softmax for probability output
    ])
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy'])
    return model

print("Training model with Cross-Entropy loss...")
model_ce = build_simple_cnn('sparse_categorical_crossentropy')
# sparse_categorical_crossentropy accepts integer labels directly
history_ce = model_ce.fit(
    x_tr, y_tr_sparse,
    epochs=10, batch_size=128,
    validation_split=0.1,
    verbose=0
)
print(f"  Cross-Entropy: final val accuracy = {max(history_ce.history['val_accuracy']):.4f}")

print("Training model with MSE loss...")
model_mse = build_simple_cnn('mean_squared_error')
# MSE needs one-hot encoded labels because it compares vectors element-wise
history_mse = model_mse.fit(
    x_tr, y_tr_oh,
    epochs=10, batch_size=128,
    validation_split=0.1,
    verbose=0
)
print(f"  MSE: final val accuracy = {max(history_mse.history['val_accuracy']):.4f}")

# Visualize the comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 11: Cross-Entropy vs MSE for Classification", fontsize=14, fontweight='bold')

axes[0].plot(history_ce.history['val_accuracy'], 'b-o', label='Cross-Entropy', markersize=4)
axes[0].plot(history_mse.history['val_accuracy'], 'r-o', label='MSE', markersize=4)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Validation Accuracy: Cross-Entropy vs MSE")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history_ce.history['val_loss'], 'b-o', label='Cross-Entropy Loss', markersize=4)
axes[1].plot(history_mse.history['val_loss'], 'r-o', label='MSE Loss', markersize=4)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Validation Loss: Cross-Entropy vs MSE")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_11_loss_comparison.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_loss_comparison.png")
print("  Observation: Cross-entropy typically converges faster for classification tasks.")


# ===========================================================
# === SECTION 3: OPTIMIZER COMPARISON (SGD vs RMSProp vs Adam) ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 3: Optimizer Comparison — SGD vs RMSProp vs Adam")
print("=" * 60)

def build_model_with_optimizer(optimizer):
    """Build a CNN compiled with the given optimizer."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

optimizers_to_compare = {
    "SGD (lr=0.01)":             keras.optimizers.SGD(learning_rate=0.01),
    "SGD+Momentum (lr=0.01)":    keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "RMSProp (lr=0.001)":        keras.optimizers.RMSprop(learning_rate=0.001),
    "Adam (lr=0.001)":           keras.optimizers.Adam(learning_rate=0.001),
}

histories = {}
for name, opt in optimizers_to_compare.items():
    print(f"  Training with {name}...")
    model = build_model_with_optimizer(opt)
    hist = model.fit(
        x_tr, y_tr_sparse,
        epochs=15, batch_size=128,
        validation_split=0.1,
        verbose=0
    )
    histories[name] = hist
    print(f"    Final val accuracy: {max(hist.history['val_accuracy']):.4f}")

# Plot the optimizer comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 11: Optimizer Comparison on MNIST", fontsize=14, fontweight='bold')

linestyles = ['-', '--', '-.', ':']
colors = ['blue', 'green', 'red', 'purple']

for (name, hist), ls, c in zip(histories.items(), linestyles, colors):
    axes[0].plot(hist.history['val_accuracy'], linestyle=ls, color=c, label=name, linewidth=2)
    axes[1].plot(hist.history['val_loss'],     linestyle=ls, color=c, label=name, linewidth=2)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Validation Accuracy by Optimizer"); axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Validation Loss by Optimizer"); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_11_optimizer_comparison.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_optimizer_comparison.png")


# ===========================================================
# === SECTION 4: LEARNING RATE EFFECTS AND SCHEDULING ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 4: Learning Rate Effects and Scheduling")
print("=" * 60)

# --- Part A: Effect of different static learning rates ---
print("Comparing different static learning rates with Adam...")
lr_values = [0.1, 0.01, 0.001, 0.0001]
lr_histories = {}

for lr in lr_values:
    model = keras.Sequential([
        layers.Conv2D(16, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(32, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    hist = model.fit(x_tr, y_tr_sparse, epochs=10, batch_size=128,
                     validation_split=0.1, verbose=0)
    lr_histories[lr] = hist
    print(f"  lr={lr}: final val acc = {max(hist.history['val_accuracy']):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 11: Effect of Learning Rate on Training", fontsize=14, fontweight='bold')

colors_lr = ['red', 'orange', 'green', 'blue']
for lr, color in zip(lr_values, colors_lr):
    h = lr_histories[lr]
    axes[0].plot(h.history['val_accuracy'], color=color, label=f"lr={lr}", linewidth=2)
    axes[1].plot(h.history['val_loss'],     color=color, label=f"lr={lr}", linewidth=2)

axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Too High / Just Right / Too Low Learning Rate"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Loss Behavior: Various Learning Rates"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_11_learning_rates.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_learning_rates.png")

# --- Part B: Learning Rate Scheduling ---
print("\nDemonstrating ReduceLROnPlateau scheduling...")

def build_model_for_scheduling():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ReduceLROnPlateau: reduce LR when val_loss doesn't improve for `patience` epochs
lr_reducer = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,          # Multiply LR by this factor on plateau
    patience=2,          # Wait this many epochs before reducing
    min_lr=1e-6,         # Minimum LR floor
    verbose=1            # Print a message when LR is reduced
)

model_scheduled = build_model_for_scheduling()
print("Training with ReduceLROnPlateau callback (initial lr=0.01):")
hist_scheduled = model_scheduled.fit(
    x_tr, y_tr_sparse,
    epochs=20, batch_size=128,
    validation_split=0.1,
    callbacks=[lr_reducer],
    verbose=0
)

# Extract learning rate history from the history object
# (LR is stored in model.optimizer.lr after training)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Lesson 11: ReduceLROnPlateau Scheduling", fontsize=14, fontweight='bold')

axes[0].plot(hist_scheduled.history['val_accuracy'], 'g-o', markersize=4, label="Scheduled LR")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Validation Accuracy")
axes[0].set_title("Accuracy with LR Scheduling"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_scheduled.history['val_loss'], 'r-o', markersize=4, label="Scheduled LR")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss")
axes[1].set_title("Loss with LR Scheduling"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_11_lr_scheduling.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_lr_scheduling.png")

# --- Part C: Visualize Binary vs Categorical Cross-Entropy values ---
print("\nVisualizing cross-entropy as a function of predicted probability...")

p = np.linspace(0.001, 0.999, 500)          # Range of predicted probabilities
bce_correct   = -np.log(p)                   # BCE when true label = 1
bce_incorrect = -np.log(1 - p)               # BCE when true label = 0
mse_correct   = (1 - p) ** 2                 # MSE when true label = 1

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(p, bce_correct,   'b-',  linewidth=2, label='Cross-Entropy: true label=1')
ax.plot(p, bce_incorrect, 'r-',  linewidth=2, label='Cross-Entropy: true label=0')
ax.plot(p, mse_correct,   'g--', linewidth=2, label='MSE: true label=1')
ax.set_xlabel("Predicted Probability p̂", fontsize=12)
ax.set_ylabel("Loss Value", fontsize=12)
ax.set_title("Lesson 11: Loss Value vs. Predicted Probability\n(Why Cross-Entropy Has Stronger Gradients)", fontsize=12)
ax.legend(fontsize=11)
ax.set_ylim(0, 5)
ax.grid(True, alpha=0.3)
ax.annotate('Confident wrong prediction:\nCross-Entropy → very high loss\nMSE → only moderate loss',
            xy=(0.05, bce_correct[10]), xytext=(0.2, 3.5),
            arrowprops=dict(arrowstyle='->', color='black'),
            fontsize=9, color='blue')

plt.tight_layout()
plt.savefig("lesson_11_loss_function_curves.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_11_loss_function_curves.png")

print("\n" + "=" * 60)
print("Lesson 11 Complete!")
print("Generated files:")
print("  - lesson_11_gradient_descent.png")
print("  - lesson_11_loss_comparison.png")
print("  - lesson_11_optimizer_comparison.png")
print("  - lesson_11_learning_rates.png")
print("  - lesson_11_lr_scheduling.png")
print("  - lesson_11_loss_function_curves.png")
print("=" * 60)
