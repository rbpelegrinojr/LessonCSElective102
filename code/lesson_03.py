"""
Lesson 3: Introduction to Neural Networks
Demonstrates neural network fundamentals from a single neuron to a full MLP.

Run: python code/lesson_03.py
Dependencies: tensorflow, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: Implementing a Single Neuron from Scratch
# =============================================================================
print("=" * 60)
print("SECTION 1: A Single Neuron from Scratch (NumPy)")
print("=" * 60)
print("\nA neuron computes: output = activation(w·x + b)")
print("where w = weights, x = inputs, b = bias\n")

def sigmoid(z):
    """Sigmoid activation: maps any real number to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-z))

def relu(z):
    """ReLU activation: max(0, z) — most common in hidden layers."""
    return np.maximum(0.0, z)

def tanh_activation(z):
    """Tanh activation: maps to (-1, 1)."""
    return np.tanh(z)

class SingleNeuron:
    def __init__(self, n_inputs, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(0, 0.1, size=n_inputs)
        self.bias = 0.0
        self.activation_name = activation
        activations = {"sigmoid": sigmoid, "relu": relu, "tanh": tanh_activation}
        self.activation = activations[activation]

    def forward(self, x):
        z = np.dot(self.weights, x) + self.bias
        output = self.activation(z)
        return z, output

    def __repr__(self):
        return (f"SingleNeuron(weights={np.round(self.weights, 3)}, "
                f"bias={self.bias:.3f}, activation={self.activation_name})")

neuron = SingleNeuron(n_inputs=3, activation="relu")
print(f"Neuron: {neuron}\n")

test_inputs = [
    np.array([1.0, 0.5, -1.0]),
    np.array([0.0, 0.0, 0.0]),
    np.array([2.0, 3.0, -2.0]),
]
print(f"{'Input':<30} {'z (pre-act)':<15} {'Output (post-act)'}")
print("-" * 60)
for x in test_inputs:
    z, out = neuron.forward(x)
    print(f"{str(np.round(x, 2)):<30} {z:<15.4f} {out:.4f}")

z_range = np.linspace(-5, 5, 200)
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("SECTION 1: Activation Functions", fontsize=13, fontweight="bold")

for ax, (name, fn, color) in zip(axes, [
    ("Sigmoid σ(z)", sigmoid, "steelblue"),
    ("ReLU max(0,z)", relu, "tomato"),
    ("Tanh tanh(z)", tanh_activation, "seagreen"),
]):
    ax.plot(z_range, fn(z_range), color=color, linewidth=2.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("z (pre-activation)")
    ax.set_ylabel("output")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 1.5)

plt.tight_layout()
plt.savefig("section1_activations.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section1_activations.png]")

# =============================================================================
# SECTION 2: 2-Layer MLP from Scratch
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: 2-Layer MLP from Scratch with Forward + Backward Pass")
print("=" * 60)
print("\nBuilding a minimal MLP to solve the XOR problem.")
print("XOR: output 1 when inputs differ, 0 when they match.\n")
print("XOR truth table:")
print("  x1  x2  |  y")
print("  ─────────────")
print("   0   0  |  0")
print("   0   1  |  1")
print("   1   0  |  1")
print("   1   1  |  0")
print("\nA single neuron cannot solve XOR (not linearly separable).")
print("We need at least one hidden layer.\n")

def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def cross_entropy_loss(y_pred, y_true):
    n = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n), y_true] + 1e-9)
    return log_likelihood.mean()

class MLP:
    def __init__(self, input_size, hidden_size, output_size, seed=0):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / input_size)
        self.W1 = rng.normal(0, scale, (input_size, hidden_size))
        self.b1 = np.zeros(hidden_size)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size))
        self.b2 = np.zeros(output_size)

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y_true, lr=0.1):
        n = X.shape[0]
        dz2 = self.a2.copy()
        dz2[range(n), y_true] -= 1
        dz2 /= n
        dW2 = self.a1.T @ dz2
        db2 = dz2.sum(axis=0)
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0).astype(float)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2

    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y_xor = np.array([0, 1, 1, 0])

mlp_scratch = MLP(input_size=2, hidden_size=4, output_size=2, seed=7)

print("Before training:")
probs = mlp_scratch.forward(X_xor)
preds = np.argmax(probs, axis=1)
print(f"  Predictions: {preds}  (expected: {y_xor})")
print(f"  Accuracy   : {(preds == y_xor).mean():.0%}\n")

losses_scratch = []
for epoch in range(2000):
    probs = mlp_scratch.forward(X_xor)
    loss = cross_entropy_loss(probs, y_xor)
    losses_scratch.append(loss)
    mlp_scratch.backward(X_xor, y_xor, lr=0.5)

print("After 2000 training steps:")
probs = mlp_scratch.forward(X_xor)
preds = mlp_scratch.predict(X_xor)
print(f"  Predictions: {preds}  (expected: {y_xor})")
print(f"  Accuracy   : {(preds == y_xor).mean():.0%}")
print(f"  Final loss : {losses_scratch[-1]:.4f}")

# =============================================================================
# SECTION 3: Building the Same Network with Keras
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Building the Same MLP with Keras Sequential API")
print("=" * 60)
print("\nKeras handles all the math for us. We just specify the architecture.\n")

import tensorflow as tf

tf.random.set_seed(42)

keras_mlp = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,), name="input"),
    tf.keras.layers.Dense(4, activation="relu", name="hidden"),
    tf.keras.layers.Dense(2, activation="softmax", name="output"),
], name="XOR_MLP")

keras_mlp.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

keras_mlp.summary()
print()

X_keras = X_xor
y_keras = y_xor

history = keras_mlp.fit(
    X_keras, y_keras,
    epochs=500,
    verbose=0,
)

print(f"Final training accuracy : {history.history['accuracy'][-1]:.0%}")
print(f"Final loss              : {history.history['loss'][-1]:.4f}")

keras_preds = np.argmax(keras_mlp.predict(X_keras, verbose=0), axis=1)
print(f"Predictions : {keras_preds}  (expected: {y_xor})")
print(f"Accuracy    : {(keras_preds == y_xor).mean():.0%}")

# =============================================================================
# SECTION 4: Visualizing the Forward Pass Step by Step
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Detailed Forward Pass Walkthrough")
print("=" * 60)
print("\nTracing exactly what happens inside the MLP for one input: x = [0, 1]\n")

x_sample = np.array([[0.0, 1.0]])
y_sample_true = 1

print(f"Input x = {x_sample[0]}")
print(f"True class = {y_sample_true} (XOR(0, 1) = 1)\n")

print("--- Layer 1: Hidden (Dense + ReLU) ---")
z1 = x_sample @ mlp_scratch.W1 + mlp_scratch.b1
print(f"  z1 = x @ W1 + b1")
print(f"  z1 = {np.round(z1[0], 4)}")
a1 = relu(z1)
print(f"  a1 = ReLU(z1) = {np.round(a1[0], 4)}")
print(f"  (ReLU zeroes out negative values)")

print("\n--- Layer 2: Output (Dense + Softmax) ---")
z2 = a1 @ mlp_scratch.W2 + mlp_scratch.b2
print(f"  z2 = a1 @ W2 + b2")
print(f"  z2 = {np.round(z2[0], 4)}")
a2 = softmax(z2)
print(f"  a2 = softmax(z2) = {np.round(a2[0], 4)}")
print(f"  Sum of probabilities: {a2[0].sum():.4f}  ← must be 1.0")
print(f"\n  Predicted class  : {np.argmax(a2[0])}")
print(f"  True class       : {y_sample_true}")
loss = cross_entropy_loss(a2, np.array([y_sample_true]))
print(f"  Cross-entropy loss: {loss:.4f}")

# =============================================================================
# SECTION 5: Training on XOR and Plotting Loss Curve
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Training on XOR Problem — Loss Curve")
print("=" * 60)
print("\nWatching the loss decrease over training reveals learning dynamics.\n")

mlp_for_plot = MLP(input_size=2, hidden_size=8, output_size=2, seed=42)
losses_for_plot = []
accuracies = []

for epoch in range(5000):
    probs = mlp_for_plot.forward(X_xor)
    loss = cross_entropy_loss(probs, y_xor)
    losses_for_plot.append(loss)
    preds = np.argmax(probs, axis=1)
    accuracies.append((preds == y_xor).mean())
    mlp_for_plot.backward(X_xor, y_xor, lr=0.3)

print(f"Epoch    0: loss={losses_for_plot[0]:.4f}, accuracy={accuracies[0]:.0%}")
print(f"Epoch  500: loss={losses_for_plot[499]:.4f}, accuracy={accuracies[499]:.0%}")
print(f"Epoch 1000: loss={losses_for_plot[999]:.4f}, accuracy={accuracies[999]:.0%}")
print(f"Epoch 2000: loss={losses_for_plot[1999]:.4f}, accuracy={accuracies[1999]:.0%}")
print(f"Epoch 5000: loss={losses_for_plot[-1]:.4f}, accuracy={accuracies[-1]:.0%}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("SECTION 5: Training Dynamics on XOR Problem", fontsize=13, fontweight="bold")

axes[0].plot(losses_scratch, color="tomato", linewidth=1.5, label="From-Scratch MLP")
axes[0].plot(history.history["loss"], color="steelblue", linewidth=1.5, label="Keras MLP")
axes[0].set_title("Loss Curves Comparison", fontsize=11)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Cross-Entropy Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale("log")

axes[1].plot(losses_for_plot, color="seagreen", linewidth=1.5)
axes[1].set_title("Loss Curve (From-Scratch, 5000 epochs)", fontsize=11)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Cross-Entropy Loss")
axes[1].grid(True, alpha=0.3)
for milestone_epoch in [500, 1000, 2000]:
    axes[1].axvline(milestone_epoch, color="orange", linestyle="--", alpha=0.7)
    axes[1].text(milestone_epoch + 30, losses_for_plot[milestone_epoch - 1] + 0.01,
                 f"e={milestone_epoch}", fontsize=8)

axes[2].plot(accuracies, color="purple", linewidth=1.5)
axes[2].set_title("Training Accuracy Over Time", fontsize=11)
axes[2].set_xlabel("Epoch")
axes[2].set_ylabel("Accuracy")
axes[2].set_ylim(-0.05, 1.05)
axes[2].axhline(1.0, color="green", linestyle="--", alpha=0.7, label="Perfect accuracy")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section5_loss_curve.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section5_loss_curve.png]")

print("\n" + "=" * 60)
print("All 5 sections complete!")
print("Generated files:")
for fname in ["section1_activations.png", "section5_loss_curve.png"]:
    print(f"  {fname}")
print("=" * 60)

plt.show()
