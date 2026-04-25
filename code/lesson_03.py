"""
Lesson 03: Introduction to Neural Networks
==========================================
This module builds a simple fully-connected (dense) neural network using
Keras and trains it on the MNIST handwritten digit dataset. We inspect
weight shapes, observe the forward pass numerically, and plot training
and validation curves to understand the learning dynamics.

Key concepts demonstrated:
  - Building a sequential model with Dense layers
  - Activation functions (ReLU, Softmax)
  - Compiling with loss function and optimiser
  - Training with model.fit() and validation split
  - Inspecting learned weights and biases
  - Plotting training history (loss and accuracy)
"""

# === STANDARD LIBRARY & THIRD-PARTY IMPORTS ===
import numpy as np                              # numerical operations
import matplotlib.pyplot as plt                 # plotting
import tensorflow as tf                        # TensorFlow backend
from tensorflow import keras                   # high-level API
from tensorflow.keras import layers            # layer building blocks
from tensorflow.keras.datasets import mnist    # MNIST dataset

# Set a random seed for reproducibility across NumPy and TensorFlow
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# === SECTION 1: DATA PREPARATION ===
# =====================================================================

print("=" * 60)
print("SECTION 1: Data Preparation")
print("=" * 60)

# Load MNIST: (60,000 train + 10,000 test) × (28×28 grayscale images)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"\nRaw data shapes:")
print(f"  x_train: {x_train.shape}, dtype: {x_train.dtype}")
print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")

# --- Step 1: Flatten 2D images into 1D vectors ---
# Dense layers expect 1D input per sample.
# 28 × 28 = 784 pixels → 784-element vector
x_train_flat = x_train.reshape(-1, 784)   # (60000, 784); -1 infers batch size
x_test_flat  = x_test.reshape(-1, 784)    # (10000, 784)

# --- Step 2: Normalise pixel values to [0.0, 1.0] ---
# Convert to float32 first (uint8 division would truncate to integers)
x_train_flat = x_train_flat.astype(np.float32) / 255.0
x_test_flat  = x_test_flat.astype(np.float32)  / 255.0

print(f"\nPrepared data shapes:")
print(f"  x_train_flat: {x_train_flat.shape}, dtype: {x_train_flat.dtype}")
print(f"  Value range:  [{x_train_flat.min():.2f}, {x_train_flat.max():.2f}]")

# Display a sample of label values
print(f"\n  First 10 training labels: {y_train[:10]}")

# =====================================================================
# === SECTION 2: BUILDING THE NEURAL NETWORK ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: Building the Neural Network")
print("=" * 60)

# Build a Sequential model — layers are stacked linearly, one after another
model = keras.Sequential([
    # Input layer — tells Keras the shape of each sample (not counting batch dim)
    keras.Input(shape=(784,)),

    # Hidden layer 1: 128 neurons with ReLU activation
    # "Dense" means every input neuron connects to every output neuron (fully connected)
    # ReLU(z) = max(0, z) — introduces non-linearity; prevents vanishing gradients
    layers.Dense(128, activation='relu', name='hidden_layer_1'),

    # Hidden layer 2: 64 neurons with ReLU activation
    # Adding depth allows the network to learn more abstract combinations of features
    layers.Dense(64, activation='relu', name='hidden_layer_2'),

    # Output layer: 10 neurons (one per digit class) with Softmax activation
    # Softmax converts raw scores (logits) into a probability distribution
    # The 10 values will always sum to 1.0; the highest is the predicted class
    layers.Dense(10, activation='softmax', name='output_layer'),
])

# Print the model architecture summary
# Shows: layer names, output shapes, and parameter counts
print("\nModel Architecture:")
model.summary()

# =====================================================================
# === SECTION 3: INSPECTING WEIGHTS AND BIASES ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: Inspecting Weights and Biases (Before Training)")
print("=" * 60)

print("\nModel has", len(model.layers), "layers:")
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()   # returns list [weight_matrix, bias_vector]
    if weights:                     # skip layers with no parameters (e.g., Input)
        W, b = weights[0], weights[1]
        n_params = W.size + b.size  # total parameters in this layer
        print(f"\nLayer {i}: {layer.name}")
        print(f"  Weight matrix shape: {W.shape}")    # (input_dim, output_dim)
        print(f"  Bias vector shape:   {b.shape}")    # (output_dim,)
        print(f"  Total parameters:    {n_params}")
        print(f"  Weight stats — min: {W.min():.4f}, max: {W.max():.4f}, "
              f"mean: {W.mean():.4f}, std: {W.std():.4f}")
        print(f"  Bias stats   — min: {b.min():.4f}, max: {b.max():.4f}, "
              f"mean: {b.mean():.4f}")

# Count total parameters manually to cross-check with model.summary()
total_params = sum(layer.count_params() for layer in model.layers)
print(f"\nTotal trainable parameters: {total_params:,}")

# Manually verify for hidden_layer_1:
# 784 inputs × 128 neurons + 128 biases = 100,352 + 128 = 100,480
print("\nManual verification for hidden_layer_1:")
print(f"  784 × 128 (weights) + 128 (biases) = {784 * 128 + 128:,}")

# =====================================================================
# === SECTION 4: COMPILING AND TRAINING ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: Compiling and Training")
print("=" * 60)

# Compile the model — specify:
#   optimizer: Adam (adaptive learning rate, generally the best default choice)
#   loss: sparse_categorical_crossentropy (for integer labels, not one-hot)
#   metrics: accuracy (fraction of correctly classified samples)
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nCompiled with:")
print("  Optimizer: Adam")
print("  Loss:      sparse_categorical_crossentropy")
print("  Metrics:   accuracy")
print("\nStarting training (15 epochs, batch_size=128, 20% validation split)...")
print("-" * 50)

# Train the model:
#   epochs=15:          process the entire training set 15 times
#   batch_size=128:     update weights after every 128-sample mini-batch
#   validation_split:   hold out 20% of training data for validation
#   verbose=1:          print progress bar for each epoch
history = model.fit(
    x_train_flat, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.2,   # 20% of 60,000 = 12,000 samples for validation
    verbose=1
)

# Evaluate on the held-out test set (never seen during training)
test_loss, test_acc = model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nTest set results:")
print(f"  Test Loss:     {test_loss:.4f}")
print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")

# =====================================================================
# === SECTION 5: VISUALISING TRAINING HISTORY ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 5: Visualising Training History")
print("=" * 60)

# history.history is a dict with keys: 'loss', 'accuracy', 'val_loss', 'val_accuracy'
print("\nAvailable history keys:", list(history.history.keys()))
print(f"Recorded over {len(history.history['loss'])} epochs")

# --- Plot 1: Training and validation loss ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dense Neural Network Training on MNIST", fontsize=14)

epochs_range = range(1, len(history.history['loss']) + 1)

ax1.plot(epochs_range, history.history['loss'],
         'b-o', markersize=4, label='Training Loss', linewidth=2)
ax1.plot(epochs_range, history.history['val_loss'],
         'r-s', markersize=4, label='Validation Loss', linewidth=2)
ax1.set_title("Loss Over Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Categorical Cross-Entropy Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(epochs_range)

# --- Plot 2: Training and validation accuracy ---
ax2.plot(epochs_range, history.history['accuracy'],
         'b-o', markersize=4, label='Training Accuracy', linewidth=2)
ax2.plot(epochs_range, history.history['val_accuracy'],
         'r-s', markersize=4, label='Validation Accuracy', linewidth=2)
ax2.axhline(y=test_acc, color='green', linestyle='--', linewidth=1.5,
            label=f'Test Accuracy: {test_acc:.3f}')
ax2.set_title("Accuracy Over Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (fraction correct)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xticks(epochs_range)
ax2.set_ylim(0.8, 1.0)   # zoom in on the interesting range

plt.tight_layout()
plt.savefig('lesson_03_training_history.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_03_training_history.png")

# --- Plot 3: Weight distribution before vs after training ---
# Extract the final weights of hidden_layer_1
W_final, b_final = model.get_layer('hidden_layer_1').get_weights()

print(f"\nFinal weights of hidden_layer_1:")
print(f"  Weight shape: {W_final.shape}")
print(f"  Bias shape:   {b_final.shape}")
print(f"  Weight stats — min: {W_final.min():.4f}, max: {W_final.max():.4f}, "
      f"std: {W_final.std():.4f}")

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.hist(W_final.flatten(), bins=80, color='steelblue', edgecolor='black',
        linewidth=0.3, alpha=0.8)
ax.set_title("Distribution of Learned Weights — Hidden Layer 1\n"
             "(After 15 epochs of training on MNIST)")
ax.set_xlabel("Weight Value")
ax.set_ylabel("Count")
ax.axvline(0, color='red', linewidth=1.5, linestyle='--', label='Zero')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lesson_03_weight_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_03_weight_distribution.png")

# --- Show predictions on a few test images ---
print("\nGenerating figure: predictions on sample test images...")

# Get model predictions: shape (10000, 10) — one probability vector per image
predictions = model.predict(x_test_flat, verbose=0)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("Dense Network Predictions on MNIST Test Images", fontsize=13)

for i in range(10):
    ax = axes[i // 5][i % 5]
    img = x_test[i]                              # original 28×28 image for display
    pred_probs = predictions[i]                  # 10 class probabilities
    pred_label = np.argmax(pred_probs)           # predicted class (argmax)
    true_label = y_test[i]                       # ground truth
    confidence = pred_probs[pred_label] * 100    # confidence percentage

    ax.imshow(img, cmap='gray', interpolation='nearest')
    colour = 'green' if pred_label == true_label else 'red'
    ax.set_title(f"True: {true_label}  Pred: {pred_label}\n{confidence:.1f}%",
                 color=colour, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_03_predictions.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_03_predictions.png")

# Final summary
print("\n" + "=" * 60)
print("LESSON 03 DEMO COMPLETE")
print("=" * 60)
print(f"\nFinal model performance:")
print(f"  Training accuracy:   {history.history['accuracy'][-1]:.4f}")
print(f"  Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"  Test accuracy:       {test_acc:.4f}")
print("\nKey takeaways from this demo:")
print("  1. A dense network with 784→128→64→10 achieves ~98% on MNIST")
print("  2. Weights start near zero (random init) and spread out during training")
print("  3. Training loss decreases; validation loss should track closely (no overfitting)")
print("  4. model.summary() shows parameter counts for every layer")
print("  5. model.get_weights() lets us inspect and visualise learned parameters")
print("\nNext: Lesson 04 — From Dense Networks to CNNs")
