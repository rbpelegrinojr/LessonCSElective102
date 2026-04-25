"""
Lesson 6: Activation Functions
CNN Image Classification Course

Sections:
  1. Plot all activation functions side by side
  2. Vanishing gradient: sigmoid vs ReLU
  3. Compare training speed with different activations
  4. Softmax demonstration
  5. Dead ReLU demonstration
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

print("=" * 60)
print("LESSON 6: Activation Functions")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Plot All Activation Functions Side by Side
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Plotting All Activation Functions ---")

x = np.linspace(-6, 6, 300)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh_fn(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x >= 0, x, alpha * x)

def elu(x, alpha=1.0):
    return np.where(x >= 0, x, alpha * (np.exp(x) - 1))

activations = {
    "Sigmoid\nσ(x) = 1/(1+e^-x)": (sigmoid(x), "steelblue"),
    "Tanh\ntanh(x)":               (tanh_fn(x), "darkorange"),
    "ReLU\nmax(0, x)":             (relu(x),    "green"),
    "Leaky ReLU\nα=0.1":           (leaky_relu(x), "red"),
    "ELU\nα=1.0":                  (elu(x),     "purple"),
}

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("Activation Functions Comparison", fontsize=14, fontweight='bold')

for ax, (name, (y_vals, color)) in zip(axes, activations.items()):
    ax.plot(x, y_vals, color=color, linewidth=2)
    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_title(name, fontsize=9)
    ax.set_xlim(-6, 6)
    ax.set_ylim(-2, 6)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x")

plt.tight_layout()
plt.savefig("section6_activations.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section6_activations.png")

print("\n  Function values at x = 0:")
for name, (y_vals, _) in activations.items():
    idx_zero = len(x) // 2
    print(f"    {name.split(chr(10))[0]:15s}: {y_vals[idx_zero]:.4f}")

print("\n  Derivative of sigmoid at x = 0:")
s0 = sigmoid(0)
print(f"    σ'(0) = σ(0)·(1 - σ(0)) = {s0:.3f} × {1-s0:.3f} = {s0*(1-s0):.4f}")
print("  (Maximum possible gradient for sigmoid is 0.25)")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Vanishing Gradient — Sigmoid vs ReLU
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Vanishing Gradient Demonstration ---")

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu_derivative(x):
    return np.where(x > 0, 1.0, 0.0)

# Simulate gradient propagation through N layers
n_layers = 20
input_val = 0.5  # typical activation value

print(f"\n  Simulating gradient magnitude through {n_layers} layers:")
print(f"  {'Layer':>6} | {'Sigmoid Gradient':>18} | {'ReLU Gradient':>15}")
print("  " + "-" * 46)

sigmoid_grad = 1.0
relu_grad = 1.0

sigmoid_grads = [1.0]
relu_grads = [1.0]

for i in range(1, n_layers + 1):
    sigmoid_grad *= sigmoid_derivative(input_val)
    relu_grad *= relu_derivative(input_val)
    sigmoid_grads.append(sigmoid_grad)
    relu_grads.append(relu_grad)
    if i in [1, 2, 5, 10, 15, 20]:
        print(f"  {i:>6} | {sigmoid_grad:>18.2e} | {relu_grad:>15.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Vanishing Gradient: Sigmoid vs ReLU", fontsize=13, fontweight='bold')

axes[0].semilogy(range(n_layers + 1), sigmoid_grads, 'b-o', markersize=4,
                 label='Sigmoid gradient')
axes[0].semilogy(range(n_layers + 1), relu_grads, 'g-s', markersize=4,
                 label='ReLU gradient')
axes[0].set_xlabel("Layer depth")
axes[0].set_ylabel("Gradient magnitude (log scale)")
axes[0].set_title("Gradient Magnitude vs Depth")
axes[0].legend()
axes[0].grid(True, alpha=0.4)

sigmoid_grads_arr = np.array(sigmoid_grads)
axes[1].plot(range(n_layers + 1), sigmoid_grads_arr, 'b-o', markersize=4,
             label='Sigmoid gradient')
axes[1].axhline(1e-7, color='red', linestyle='--', label='Effective zero (1e-7)')
axes[1].set_xlabel("Layer depth")
axes[1].set_ylabel("Gradient magnitude (linear scale)")
axes[1].set_title("Sigmoid Gradient (linear scale)")
axes[1].legend()
axes[1].grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig("section6_vanishing_grad.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section6_vanishing_grad.png")
print(f"\n  After {n_layers} layers, sigmoid gradient ≈ {sigmoid_grads[-1]:.2e}")
print("  This is why deep networks with sigmoid fail to train!")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Compare Training Speed With Different Activations
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Training Speed Comparison ---")

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test.reshape(-1, 784).astype('float32')  / 255.0

# Use a small subset for quick comparison
X_sub = X_train[:10000]
y_sub = y_train[:10000]

def build_model(activation):
    model = models.Sequential([
        layers.Dense(128, activation=activation, input_shape=(784,)),
        layers.Dense(64,  activation=activation),
        layers.Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

activation_configs = ['sigmoid', 'tanh', 'relu']
histories_act = {}

print("  Training 3-layer MLP on 10,000 MNIST samples for 5 epochs each...")
for act in activation_configs:
    print(f"    Activation: {act.upper():<10}", end="", flush=True)
    model = build_model(act)
    hist = model.fit(X_sub, y_sub, epochs=5, batch_size=64,
                     validation_split=0.1, verbose=0)
    histories_act[act] = hist.history
    final_acc = hist.history['val_accuracy'][-1]
    print(f" → Val accuracy after 5 epochs: {final_acc:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
colors = {'sigmoid': 'blue', 'tanh': 'orange', 'relu': 'green'}

for act, hist in histories_act.items():
    axes[0].plot(hist['loss'], label=act, color=colors[act])
    axes[1].plot(hist['val_accuracy'], label=act, color=colors[act])

axes[0].set_title("Training Loss by Activation Function")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].set_title("Validation Accuracy by Activation Function")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section6_training_comparison.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section6_training_comparison.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Softmax Demonstration
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Softmax Demonstration ---")

def softmax(logits):
    """Numerically stable softmax."""
    exp_shifted = np.exp(logits - np.max(logits))
    return exp_shifted / exp_shifted.sum()

class_names = ['airplane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

example_logits = np.array([1.2, 0.5, -0.3, 3.1, 0.2,
                            -0.8, 0.0, 1.5, 2.2, -0.1])

probs = softmax(example_logits)

print("  Raw logits (model output):")
for name, logit in zip(class_names, example_logits):
    print(f"    {name:10s}: {logit:6.2f}")

print("\n  Softmax probabilities:")
for name, prob in zip(class_names, probs):
    bar = "█" * int(prob * 40)
    print(f"    {name:10s}: {prob:.4f}  {bar}")

print(f"\n  Probabilities sum to: {probs.sum():.6f}")
print(f"  Predicted class: {class_names[np.argmax(probs)]} "
      f"(confidence: {probs.max():.2%})")

print("\n  Effect of scaling logits (sharper/softer predictions):")
for scale in [0.5, 1.0, 2.0, 5.0]:
    scaled_probs = softmax(example_logits * scale)
    print(f"    Scale={scale:.1f}: max prob = {scaled_probs.max():.4f}, "
          f"entropy = {-np.sum(scaled_probs * np.log(scaled_probs + 1e-9)):.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].bar(class_names, example_logits, color='steelblue', edgecolor='black')
axes[0].set_title("Raw Logits")
axes[0].set_ylabel("Logit value")
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, axis='y', alpha=0.3)

axes[1].bar(class_names, probs, color='green', edgecolor='black')
axes[1].set_title("Softmax Probabilities (sum = 1)")
axes[1].set_ylabel("Probability")
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("section6_softmax.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section6_softmax.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Dead ReLU Demonstration
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Dead ReLU Demonstration ---")

np.random.seed(42)

def count_dead_neurons(weights_and_biases, X_sample):
    """Count neurons that output 0 for all inputs in a sample."""
    activations_out = X_sample
    dead_counts = []
    for W, b in weights_and_biases:
        pre_act = activations_out @ W + b
        activations_out = np.maximum(0, pre_act)  # ReLU
        dead = np.mean(activations_out == 0, axis=0)
        dead_counts.append(np.mean(dead))
    return dead_counts

X_sample = np.random.randn(200, 784).astype('float32')

print("  Testing with different weight initializations (simulating high learning rate):")
print(f"  {'Init Scale':>12} | {'Layer 1 Dead %':>16} | {'Layer 2 Dead %':>16}")
print("  " + "-" * 52)

for scale in [0.01, 0.1, 1.0, 5.0, 10.0]:
    W1 = np.random.randn(784, 128) * scale
    b1 = np.ones(128) * (-scale * 2)  # bias shifted negative → kills neurons
    W2 = np.random.randn(128, 64) * scale
    b2 = np.ones(64) * (-scale * 2)
    dead = count_dead_neurons([(W1, b1), (W2, b2)], X_sample)
    print(f"  {scale:>12.2f} | {dead[0]:>15.1%} | {dead[1]:>15.1%}")

print("\n  Comparing ReLU vs Leaky ReLU dead neurons (high negative bias):")
n_neurons = 256
W = np.random.randn(784, n_neurons)
b = np.full(n_neurons, -5.0)  # very negative bias — kills many ReLU neurons
pre_act = X_sample @ W + b

relu_activations   = np.maximum(0, pre_act)
leaky_activations  = np.where(pre_act >= 0, pre_act, 0.01 * pre_act)

relu_dead   = np.mean(relu_activations == 0, axis=0)
leaky_dead  = np.mean(leaky_activations == 0, axis=0)

print(f"    ReLU dead neurons:        {relu_dead.mean():.1%} of {n_neurons} neurons")
print(f"    Leaky ReLU dead neurons:  {leaky_dead.mean():.1%} of {n_neurons} neurons")
print("    (Leaky ReLU never truly dies — small negative slope preserves gradient)")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(relu_dead, bins=20, color='red', edgecolor='black', alpha=0.7)
axes[0].set_title("ReLU: Fraction of Inputs = 0 per Neuron")
axes[0].set_xlabel("Fraction of inputs giving zero output")
axes[0].set_ylabel("Number of neurons")
axes[0].axvline(relu_dead.mean(), color='darkred', linestyle='--',
                label=f'Mean = {relu_dead.mean():.2%}')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].hist(leaky_dead, bins=20, color='green', edgecolor='black', alpha=0.7)
axes[1].set_title("Leaky ReLU: Fraction of Inputs = 0 per Neuron")
axes[1].set_xlabel("Fraction of inputs giving zero output")
axes[1].set_ylabel("Number of neurons")
axes[1].axvline(leaky_dead.mean(), color='darkgreen', linestyle='--',
                label=f'Mean = {leaky_dead.mean():.2%}')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section6_dead_relu.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section6_dead_relu.png")

print("\n" + "=" * 60)
print("Lesson 6 Complete!")
print("Generated images:")
print("  section6_activations.png")
print("  section6_vanishing_grad.png")
print("  section6_training_comparison.png")
print("  section6_softmax.png")
print("  section6_dead_relu.png")
print("=" * 60)
