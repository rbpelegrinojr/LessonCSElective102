"""
Lesson 06: Activation Functions
================================
This module demonstrates the activation functions used in neural networks:
ReLU, Leaky ReLU, ELU, sigmoid, tanh, and softmax. It visualizes each
function, shows their derivatives, demonstrates the dying ReLU problem,
and compares training with ReLU vs sigmoid activations on MNIST.
"""

# === IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("=" * 60)


# =============================================================================
# === SECTION 1: ACTIVATION FUNCTION DEFINITIONS AND PLOTS ===
# =============================================================================

print("\n[SECTION 1] Plotting all activation functions and their derivatives")
print("-" * 60)

# Create a range of input values from -5 to 5
x = np.linspace(-5, 5, 500)

# --- Define all activation functions in numpy ---

def relu(x):
    """ReLU: max(0, x) — most common hidden-layer activation"""
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    """Leaky ReLU: allows small negative slope to prevent dying neurons"""
    return np.where(x > 0, x, alpha * x)

def elu(x, alpha=1.0):
    """ELU: exponential for negative inputs; smooth negative saturation"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def sigmoid(x):
    """Sigmoid: squashes input to (0, 1); used in binary output layers"""
    return 1.0 / (1.0 + np.exp(-x))

def tanh_fn(x):
    """Tanh: squashes input to (-1, 1); zero-centered version of sigmoid"""
    return np.tanh(x)

# --- Define derivatives (gradients) of each function ---

def relu_grad(x):
    """Gradient of ReLU: 1 if x > 0, else 0"""
    return np.where(x > 0, 1.0, 0.0)

def leaky_relu_grad(x, alpha=0.1):
    """Gradient of Leaky ReLU: 1 if x > 0, else alpha"""
    return np.where(x > 0, 1.0, alpha)

def sigmoid_grad(x):
    """Gradient of sigmoid: sigmoid(x) * (1 - sigmoid(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_grad(x):
    """Gradient of tanh: 1 - tanh(x)^2"""
    return 1.0 - np.tanh(x) ** 2

# --- Create a comprehensive visualization ---

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Lesson 06: Activation Functions and Their Derivatives',
             fontsize=16, fontweight='bold')

activations = [
    ("ReLU", relu(x), relu_grad(x), 'steelblue'),
    ("Leaky ReLU (α=0.1)", leaky_relu(x), leaky_relu_grad(x), 'darkorange'),
    ("ELU (α=1.0)", elu(x), None, 'green'),
    ("Sigmoid", sigmoid(x), sigmoid_grad(x), 'crimson'),
    ("Tanh", tanh_fn(x), tanh_grad(x), 'purple'),
]

# Plot function value and gradient for each activation
for idx, (name, y_vals, grad_vals, color) in enumerate(activations):
    row, col = divmod(idx, 3)
    ax = axes[row][col]

    ax.plot(x, y_vals, color=color, linewidth=2.5, label='f(x)')
    if grad_vals is not None:
        # Plot gradient as dashed line
        ax.plot(x, grad_vals, color=color, linewidth=1.5,
                linestyle='--', alpha=0.7, label="f'(x)")

    ax.axhline(y=0, color='black', linewidth=0.8, linestyle='-')  # y=0 reference
    ax.axvline(x=0, color='black', linewidth=0.8, linestyle='-')  # x=0 reference
    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_xlabel('Input x')
    ax.set_ylabel('Output')
    ax.legend(fontsize=10)
    ax.set_ylim(-2, 3)
    ax.grid(True, alpha=0.3)

# Last subplot: overlay all functions for comparison
ax_last = axes[1][2]
ax_last.plot(x, relu(x), label='ReLU', linewidth=2)
ax_last.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2, linestyle='--')
ax_last.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
ax_last.plot(x, tanh_fn(x), label='Tanh', linewidth=2)
ax_last.axhline(y=0, color='black', linewidth=0.8)
ax_last.axvline(x=0, color='black', linewidth=0.8)
ax_last.set_title('All Functions — Comparison', fontsize=13, fontweight='bold')
ax_last.set_xlabel('Input x')
ax_last.set_ylabel('Output')
ax_last.legend(fontsize=9)
ax_last.set_ylim(-2, 3)
ax_last.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/lesson_06_activations.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_06_activations.png")
plt.close()


# =============================================================================
# === SECTION 2: SOFTMAX DEMONSTRATION ===
# =============================================================================

print("\n[SECTION 2] Demonstrating Softmax for multi-class output")
print("-" * 60)

def softmax(logits):
    """
    Softmax converts raw logit scores to a probability distribution.
    Subtracting the max improves numerical stability (no large exp values).
    """
    exp_scores = np.exp(logits - np.max(logits))  # subtract max for stability
    return exp_scores / exp_scores.sum()

# Example: a 4-class classification problem
logit_examples = [
    np.array([3.0, 1.0, 0.2, -1.5]),   # clear winner: class 0
    np.array([1.0, 1.0, 1.0,  1.0]),   # uniform logits: equal probabilities
    np.array([0.5, 0.4, 0.3,  0.2]),   # close competition
]

for i, logits in enumerate(logit_examples):
    probs = softmax(logits)
    predicted_class = np.argmax(probs)
    print(f"  Example {i+1}:")
    print(f"    Logits:       {logits}")
    print(f"    Probabilities:{np.round(probs, 4)}")
    print(f"    Sum of probs: {probs.sum():.6f} (should be exactly 1.0)")
    print(f"    Predicted class: {predicted_class}")
    print()

# Visualize softmax output for example 1
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle('Lesson 06: Softmax Transformation', fontsize=14, fontweight='bold')

logits_demo = np.array([3.0, 1.0, 0.2, -1.5])
probs_demo = softmax(logits_demo)
classes = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

# Plot raw logits
ax1.bar(classes, logits_demo, color=['gold', 'skyblue', 'skyblue', 'skyblue'],
        edgecolor='black')
ax1.set_title('Raw Logits (before softmax)', fontsize=12)
ax1.set_ylabel('Score')
ax1.axhline(y=0, color='black', linewidth=0.8)

# Plot softmax probabilities
ax2.bar(classes, probs_demo, color=['gold', 'skyblue', 'skyblue', 'skyblue'],
        edgecolor='black')
ax2.set_title('After Softmax (probabilities sum to 1)', fontsize=12)
ax2.set_ylabel('Probability')
for j, p in enumerate(probs_demo):
    ax2.text(j, p + 0.01, f'{p:.3f}', ha='center', fontsize=10)
ax2.set_ylim(0, 0.75)

plt.tight_layout()
plt.savefig('code/lesson_06_softmax.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_06_softmax.png")
plt.close()


# =============================================================================
# === SECTION 3: DYING RELU DEMONSTRATION ===
# =============================================================================

print("\n[SECTION 3] Demonstrating the Dying ReLU Problem")
print("-" * 60)

# Simulate a layer of neurons with negative bias (simulating dying ReLU scenario)
np.random.seed(42)

# Create synthetic pre-activation values (inputs to ReLU)
# Simulating a batch of inputs after a layer with very negative weights/biases
pre_activation_normal = np.random.randn(1000)           # centered around 0
pre_activation_biased = np.random.randn(1000) - 3.0     # shifted very negative

# Apply ReLU to both
relu_normal = relu(pre_activation_normal)
relu_biased = relu(pre_activation_biased)

# Count dead neurons (those that output 0)
dead_normal = np.sum(relu_normal == 0) / len(relu_normal) * 100
dead_biased = np.sum(relu_biased == 0) / len(relu_biased) * 100

print(f"  Normal activations  → Dead neurons (output=0): {dead_normal:.1f}%")
print(f"  Biased activations  → Dead neurons (output=0): {dead_biased:.1f}%")

# Compare ReLU vs Leaky ReLU on biased inputs
leaky_biased = leaky_relu(pre_activation_biased)
dead_leaky = np.sum(leaky_biased == 0) / len(leaky_biased) * 100
print(f"  Leaky ReLU on biased → Dead neurons:          {dead_leaky:.1f}%")
print("  → Leaky ReLU eliminates dying neuron problem!")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Lesson 06: Dying ReLU Problem', fontsize=14, fontweight='bold')

# Histogram of normal pre-activations
axes[0].hist(pre_activation_normal, bins=40, color='steelblue', alpha=0.7)
axes[0].axvline(x=0, color='red', linewidth=2, label='ReLU threshold')
axes[0].set_title('Normal Pre-activations\n(ReLU: ~50% active)', fontsize=11)
axes[0].set_xlabel('Pre-activation value')
axes[0].legend()

# Histogram of biased pre-activations (dying ReLU)
axes[1].hist(pre_activation_biased, bins=40, color='crimson', alpha=0.7)
axes[1].axvline(x=0, color='red', linewidth=2, label='ReLU threshold')
axes[1].set_title(f'Dying ReLU: {dead_biased:.0f}% neurons dead\n'
                  '(always-negative inputs)', fontsize=11)
axes[1].set_xlabel('Pre-activation value')
axes[1].legend()

# Compare ReLU vs Leaky ReLU output distribution
axes[2].hist(relu_biased, bins=40, color='crimson', alpha=0.6, label='ReLU output')
axes[2].hist(leaky_biased, bins=40, color='green', alpha=0.6, label='Leaky ReLU output')
axes[2].set_title('ReLU vs Leaky ReLU Output\n(on dying-ReLU scenario)', fontsize=11)
axes[2].set_xlabel('Activation output')
axes[2].legend()

plt.tight_layout()
plt.savefig('code/lesson_06_dying_relu.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_06_dying_relu.png")
plt.close()


# =============================================================================
# === SECTION 4: COMPARE RELU VS SIGMOID TRAINING ON MNIST ===
# =============================================================================

print("\n[SECTION 4] Training comparison: ReLU vs Sigmoid hidden layers on MNIST")
print("-" * 60)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixel values to [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32')  / 255.0

# Flatten 28x28 images to 784-dim vectors for this dense-only comparison
x_train_flat = x_train.reshape(-1, 784)
x_test_flat  = x_test.reshape(-1, 784)

# Use a small subset for faster demonstration
SUBSET = 10000
x_sub = x_train_flat[:SUBSET]
y_sub = y_train[:SUBSET]

print(f"  Training on {SUBSET} MNIST samples (subset for speed)")
print(f"  Input shape: {x_sub.shape}")

def build_model(activation_name):
    """Build a simple 3-layer dense network with the given activation function."""
    model = keras.Sequential([
        keras.layers.Dense(256, activation=activation_name, input_shape=(784,)),
        keras.layers.Dense(128, activation=activation_name),
        keras.layers.Dense(64,  activation=activation_name),
        keras.layers.Dense(10,  activation='softmax'),   # always softmax at output
    ], name=f'model_{activation_name}')

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

EPOCHS = 15
histories = {}

for act_fn in ['relu', 'sigmoid']:
    print(f"\n  Training with '{act_fn}' activation...")
    model = build_model(act_fn)
    history = model.fit(
        x_sub, y_sub,
        epochs=EPOCHS,
        batch_size=64,
        validation_split=0.1,
        verbose=0   # suppress per-epoch output for cleanliness
    )
    histories[act_fn] = history
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"  Final validation accuracy ({act_fn}): {final_val_acc:.4f}")

# Plot training comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Lesson 06: ReLU vs Sigmoid — Training on MNIST',
             fontsize=14, fontweight='bold')

colors = {'relu': 'steelblue', 'sigmoid': 'crimson'}

for act_fn, hist in histories.items():
    color = colors[act_fn]
    epochs_range = range(1, EPOCHS + 1)

    # Training loss
    ax1.plot(epochs_range, hist.history['loss'],
             color=color, linewidth=2, label=f'{act_fn} (train)')
    ax1.plot(epochs_range, hist.history['val_loss'],
             color=color, linewidth=2, linestyle='--', label=f'{act_fn} (val)')

    # Validation accuracy
    ax2.plot(epochs_range, hist.history['val_accuracy'],
             color=color, linewidth=2, label=f'{act_fn}')

ax1.set_title('Training & Validation Loss', fontsize=12)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_title('Validation Accuracy', fontsize=12)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('code/lesson_06_relu_vs_sigmoid.png', dpi=120, bbox_inches='tight')
print("\n  Saved: code/lesson_06_relu_vs_sigmoid.png")
plt.close()

print("\n" + "=" * 60)
print("Lesson 06 complete! Generated files:")
print("  - code/lesson_06_activations.png")
print("  - code/lesson_06_softmax.png")
print("  - code/lesson_06_dying_relu.png")
print("  - code/lesson_06_relu_vs_sigmoid.png")
print("Key takeaways:")
print("  1. Without non-linearity, stacked layers = one linear layer")
print("  2. ReLU is fast and avoids vanishing gradients for positive inputs")
print("  3. Sigmoid/tanh saturate, causing vanishing gradients in deep nets")
print("  4. Dying ReLU occurs when neurons always receive negative inputs")
print("  5. Leaky ReLU fixes dying ReLU with a small negative slope")
