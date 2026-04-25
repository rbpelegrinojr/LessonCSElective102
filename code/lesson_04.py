"""
Lesson 04: From Dense Networks to CNNs
========================================
This module directly compares a fully-connected (Dense) network against a
Convolutional Neural Network (CNN) on MNIST. We count parameters for each
architecture, train both models, compare their accuracies, and demonstrate
the translation invariance advantage of CNNs with a practical shift experiment.

Key concepts demonstrated:
  - Parameter explosion in dense networks for large inputs
  - Parameter counting for Conv2D and Dense layers
  - Side-by-side accuracy comparison: Dense vs CNN
  - Translation invariance test: shifted digit classification
  - Feature map visualisation from the first convolutional layer
"""

# === STANDARD LIBRARY & THIRD-PARTY IMPORTS ===
import numpy as np                              # numerical operations
import matplotlib.pyplot as plt                 # visualisation
import tensorflow as tf                        # TensorFlow backend
from tensorflow import keras                   # high-level API
from tensorflow.keras import layers            # layer building blocks
from tensorflow.keras.datasets import mnist    # MNIST dataset

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# === SECTION 1: PARAMETER COUNT ANALYSIS ===
# =====================================================================

print("=" * 60)
print("SECTION 1: Parameter Count Analysis")
print("=" * 60)

print("\nCalculating parameter counts for dense layers on different image sizes:")
print("-" * 65)
print(f"{'Image Size':<25} {'Input Neurons':>15} {'Dense(512) Params':>20}")
print("-" * 65)

image_configs = [
    ("MNIST 28×28 gray",       28,   28,  1),
    ("CIFAR-10 32×32 RGB",     32,   32,  3),
    ("VGA 640×480 RGB",       640,  480,  3),
    ("HD 1280×720 RGB",      1280,  720,  3),
    ("4K 3840×2160 RGB",     3840, 2160,  3),
]

for name, h, w, c in image_configs:
    input_neurons = h * w * c
    dense_params  = input_neurons * 512 + 512  # weights + biases for one Dense(512)
    print(f"{name:<25} {input_neurons:>15,}  {dense_params:>19,}")

print("\n→ Dense networks become impractical above ~64×64 images.")
print("  CNNs solve this with parameter sharing.\n")

# Compare Dense vs CNN parameter counts for MNIST (28×28 gray)
print("\nDetailed comparison: MNIST (28×28 grayscale, flattened to 784)")
print("-" * 55)
print(f"{'Configuration':<35} {'Parameters':>15}")
print("-" * 55)

# Dense layers
dense_config = [
    ("Dense(784→256) + bias", 784 * 256 + 256),
    ("Dense(256→128) + bias", 256 * 128 + 128),
    ("Dense(128→10)  + bias", 128 * 10  + 10),
]
dense_total = sum(p for _, p in dense_config)
for name, p in dense_config:
    print(f"  {name:<33} {p:>15,}")
print(f"  {'Total (Dense Network)':<33} {dense_total:>15,}")

print()

# CNN layers
cnn_config = [
    ("Conv2D(1→32, 3×3) + bias", (3 * 3 * 1 + 1) * 32),
    ("Conv2D(32→64, 3×3) + bias", (3 * 3 * 32 + 1) * 64),
    ("Dense(64×7×7→64) + bias",   64 * 7 * 7 * 64 + 64),   # after two 2×2 maxpools
    ("Dense(64→10) + bias",       64 * 10 + 10),
]
cnn_total = sum(p for _, p in cnn_config)
for name, p in cnn_config:
    print(f"  {name:<33} {p:>15,}")
print(f"  {'Total (CNN)':<33} {cnn_total:>15,}")
print(f"\n  CNN uses {dense_total / cnn_total:.1f}× fewer parameters than Dense")

# =====================================================================
# === SECTION 2: DATA PREPARATION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: Data Preparation")
print("=" * 60)

# Load MNIST
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# --- Prepare data for DENSE network (flattened, normalised) ---
x_train_flat = x_train.reshape(-1, 784).astype(np.float32) / 255.0
x_test_flat  = x_test.reshape(-1, 784).astype(np.float32)  / 255.0

# --- Prepare data for CNN (4D tensor: N × H × W × C) ---
# Keras CNNs require channels-last format: (batch, height, width, channels)
# Since MNIST is grayscale (1 channel), we add a channel dimension with [..., np.newaxis]
x_train_cnn = x_train[..., np.newaxis].astype(np.float32) / 255.0   # (60000, 28, 28, 1)
x_test_cnn  = x_test[..., np.newaxis].astype(np.float32)  / 255.0   # (10000, 28, 28, 1)

print(f"\nDense network input shape:  {x_train_flat.shape}")   # (60000, 784)
print(f"CNN input shape:            {x_train_cnn.shape}")     # (60000, 28, 28, 1)

# =====================================================================
# === SECTION 3: BUILDING AND TRAINING BOTH MODELS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: Building and Training Both Models")
print("=" * 60)

# --- Dense Network ---
print("\n--- Dense Network Architecture ---")
dense_model = keras.Sequential([
    keras.Input(shape=(784,)),
    layers.Dense(256, activation='relu', name='dense_h1'),   # hidden layer 1
    layers.Dense(128, activation='relu', name='dense_h2'),   # hidden layer 2
    layers.Dense(10,  activation='softmax', name='dense_out'),  # output
], name='DenseNetwork')
dense_model.summary()

dense_model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

print("\nTraining Dense Network (10 epochs)...")
dense_history = dense_model.fit(
    x_train_flat, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,   # 10% validation
    verbose=1
)

dense_test_loss, dense_test_acc = dense_model.evaluate(x_test_flat, y_test, verbose=0)
print(f"\nDense Network — Test Accuracy: {dense_test_acc:.4f} ({dense_test_acc*100:.2f}%)")

# --- CNN ---
print("\n--- CNN Architecture ---")
cnn_model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),

    # First convolutional block: 32 filters, 3×3, same padding preserves 28×28 dims
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv1'),
    # MaxPooling halves spatial dimensions: 28×28 → 14×14
    layers.MaxPooling2D(pool_size=(2, 2), name='pool1'),

    # Second convolutional block: 64 filters, deeper features
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv2'),
    # MaxPooling halves again: 14×14 → 7×7
    layers.MaxPooling2D(pool_size=(2, 2), name='pool2'),

    # Flatten converts (7, 7, 64) tensor to a 1D vector of length 3136
    layers.Flatten(name='flatten'),

    # Dense head for final classification
    layers.Dense(64, activation='relu', name='dense_head'),
    layers.Dense(10, activation='softmax', name='cnn_out'),
], name='CNN')
cnn_model.summary()

cnn_model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

print("\nTraining CNN (10 epochs)...")
cnn_history = cnn_model.fit(
    x_train_cnn, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

cnn_test_loss, cnn_test_acc = cnn_model.evaluate(x_test_cnn, y_test, verbose=0)
print(f"\nCNN — Test Accuracy: {cnn_test_acc:.4f} ({cnn_test_acc*100:.2f}%)")

# =====================================================================
# === SECTION 4: COMPARISON VISUALISATIONS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: Comparison and Translation Invariance Demo")
print("=" * 60)

# --- Plot 1: Accuracy comparison over epochs ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Dense Network vs CNN on MNIST", fontsize=14)

epochs = range(1, 11)

# Training accuracy comparison
axes[0].plot(epochs, dense_history.history['accuracy'],
             'b-o', markersize=5, label='Dense Train', linewidth=2)
axes[0].plot(epochs, dense_history.history['val_accuracy'],
             'b--s', markersize=5, label='Dense Val', linewidth=1.5, alpha=0.7)
axes[0].plot(epochs, cnn_history.history['accuracy'],
             'r-o', markersize=5, label='CNN Train', linewidth=2)
axes[0].plot(epochs, cnn_history.history['val_accuracy'],
             'r--s', markersize=5, label='CNN Val', linewidth=1.5, alpha=0.7)
axes[0].set_title("Accuracy Over Epochs")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim(0.9, 1.0)

# Parameter count comparison (bar chart)
model_names  = ['Dense\nNetwork', 'CNN']
param_counts = [dense_model.count_params(), cnn_model.count_params()]
bar_colours  = ['steelblue', 'tomato']

bars = axes[1].bar(model_names, param_counts, color=bar_colours, edgecolor='black', width=0.5)
axes[1].set_title("Total Parameter Count")
axes[1].set_ylabel("Number of Parameters")

# Annotate bars with exact counts
for bar, count in zip(bars, param_counts):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                 f"{count:,}", ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('lesson_04_comparison.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_04_comparison.png")

# Print summary table
print("\nFinal Results Summary:")
print(f"{'Model':<20} {'Parameters':>15} {'Test Accuracy':>15}")
print("-" * 52)
print(f"{'Dense Network':<20} {dense_model.count_params():>15,} {dense_test_acc:>14.4f}")
print(f"{'CNN':<20} {cnn_model.count_params():>15,} {cnn_test_acc:>14.4f}")

# --- Translation Invariance Demo ---
print("\n--- Translation Invariance Experiment ---")
print("Testing how each model handles digit images shifted horizontally...")

# Get a test image of a specific digit
test_idx = 0
original_img = x_test[test_idx]   # shape (28, 28), dtype uint8
true_label   = y_test[test_idx]

def shift_image_right(img, pixels):
    """Shift a 28×28 image to the right by `pixels` columns, fill left with zeros."""
    shifted = np.zeros_like(img)
    if pixels < 28:
        shifted[:, pixels:] = img[:, :28 - pixels]   # copy original to right portion
    return shifted

# Test with different amounts of rightward shifting
shift_amounts = [0, 3, 6, 9, 12]
dense_preds = []
cnn_preds   = []
confidences_dense = []
confidences_cnn   = []

for shift in shift_amounts:
    shifted_img = shift_image_right(original_img, shift)

    # Prepare for dense model (flattened)
    dense_input = shifted_img.reshape(1, 784).astype(np.float32) / 255.0
    dense_pred_probs = dense_model.predict(dense_input, verbose=0)[0]
    dense_pred = np.argmax(dense_pred_probs)
    dense_preds.append(dense_pred)
    confidences_dense.append(dense_pred_probs[true_label] * 100)

    # Prepare for CNN (4D tensor)
    cnn_input = shifted_img[np.newaxis, ..., np.newaxis].astype(np.float32) / 255.0
    cnn_pred_probs = cnn_model.predict(cnn_input, verbose=0)[0]
    cnn_pred = np.argmax(cnn_pred_probs)
    cnn_preds.append(cnn_pred)
    confidences_cnn.append(cnn_pred_probs[true_label] * 100)

print(f"\nTrue label: {true_label}")
print(f"\n{'Shift':<10} {'Dense Pred':>12} {'Dense Conf%':>12} {'CNN Pred':>10} {'CNN Conf%':>12}")
print("-" * 58)
for shift, dp, dc, cp, cc in zip(shift_amounts, dense_preds, confidences_dense,
                                  cnn_preds, confidences_cnn):
    print(f"{shift:>3} pixels  {dp:>12}  {dc:>10.1f}%  {cp:>10}  {cc:>10.1f}%")

# Visualise the shifted images
fig, axes = plt.subplots(2, len(shift_amounts), figsize=(16, 6))
fig.suptitle(f"Translation Invariance Test — True Digit: {true_label}", fontsize=13)

for col, shift in enumerate(shift_amounts):
    shifted = shift_image_right(original_img, shift)

    axes[0, col].imshow(shifted, cmap='gray', interpolation='nearest')
    axes[0, col].set_title(f"Shift: {shift}px", fontsize=9)
    axes[0, col].axis('off')

    # Show confidence bars for Dense and CNN
    axes[1, col].bar(['Dense', 'CNN'],
                     [confidences_dense[col], confidences_cnn[col]],
                     color=['steelblue', 'tomato'], edgecolor='black')
    axes[1, col].set_ylim(0, 105)
    axes[1, col].set_ylabel("Confidence %" if col == 0 else "")
    axes[1, col].set_title(f"D:{dense_preds[col]} C:{cnn_preds[col]}", fontsize=9)
    axes[1, col].axhline(50, color='gray', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('lesson_04_translation_test.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_04_translation_test.png")

# Final summary
print("\n" + "=" * 60)
print("LESSON 04 DEMO COMPLETE")
print("=" * 60)
print("\nKey takeaways from this demo:")
print("  1. Dense networks require vastly more parameters for the same task")
print("  2. CNNs achieve higher accuracy with fewer parameters")
print("  3. CNNs are more robust to spatial shifts of the input pattern")
print("  4. Parameter sharing is the key efficiency mechanism of CNNs")
print("  5. Pooling layers provide additional spatial invariance")
print("\nNext: Lesson 05 — The Convolution Operation Deep Dive")
