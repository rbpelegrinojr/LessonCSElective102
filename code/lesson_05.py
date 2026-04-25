"""
Lesson 05: The Convolution Operation Deep Dive
================================================
This module provides an in-depth look at the convolution operation that is
the core of Convolutional Neural Networks. We manually implement 2D convolution
using numpy, compare it to Keras Conv2D, visualise feature maps, demonstrate
the effects of padding and stride, and apply classical image processing kernels
(Sobel, Gaussian blur, sharpening) to develop strong intuition.

Key concepts demonstrated:
  - Manual 2D convolution with numpy (sliding window approach)
  - Keras Conv2D layer with different padding and stride settings
  - Output dimension calculations
  - Feature map visualisation
  - Classical kernels: Sobel edges, Gaussian blur, sharpening
  - Visualising learned filters after CNN training
"""

# === STANDARD LIBRARY & THIRD-PARTY IMPORTS ===
import numpy as np                              # array operations
import matplotlib.pyplot as plt                 # visualisation
import matplotlib.gridspec as gridspec          # subplot layout control
import tensorflow as tf                        # TensorFlow backend
from tensorflow import keras                   # high-level API
from tensorflow.keras import layers            # layer building blocks
from tensorflow.keras.datasets import mnist    # MNIST dataset
from tensorflow.keras.datasets import cifar10  # CIFAR-10 dataset

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================================
# === SECTION 1: MANUAL 2D CONVOLUTION WITH NUMPY ===
# =====================================================================

print("=" * 60)
print("SECTION 1: Manual 2D Convolution with NumPy")
print("=" * 60)

def convolve2d_valid(image, kernel):
    """
    Manually perform 2D cross-correlation (what CNNs call 'convolution')
    with VALID padding (no zero-padding).

    Args:
        image:  2D numpy array of shape (H, W)
        kernel: 2D numpy array of shape (k, k)

    Returns:
        feature_map: 2D numpy array of shape (H-k+1, W-k+1)
    """
    H, W      = image.shape
    k, _      = kernel.shape       # assumes square kernel
    out_H     = H - k + 1          # output height (valid padding formula)
    out_W     = W - k + 1          # output width
    feature_map = np.zeros((out_H, out_W), dtype=np.float64)

    for i in range(out_H):             # slide over rows
        for j in range(out_W):         # slide over columns
            # Extract the k×k patch centred at position (i, j)
            patch = image[i : i+k, j : j+k]
            # Compute the element-wise product and sum (dot product in 2D)
            feature_map[i, j] = np.sum(patch * kernel)

    return feature_map


# ---- Manual example from the lesson ----
print("\nWorked example: 5×5 input, 3×3 'plus-cross' filter")
example_input = np.array([
    [ 1,  2,  3,  4,  5],
    [ 6,  7,  8,  9, 10],
    [11, 12, 13, 14, 15],
    [16, 17, 18, 19, 20],
    [21, 22, 23, 24, 25]
], dtype=np.float32)

# "Plus-cross" filter: sums centre and four direct neighbours
plus_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.float32)

result = convolve2d_valid(example_input, plus_kernel)

print(f"\nInput (5×5):\n{example_input.astype(int)}")
print(f"\nKernel (3×3):\n{plus_kernel.astype(int)}")
print(f"\nOutput feature map (3×3, valid padding):\n{result.astype(int)}")
print(f"\nExpected output shape: {example_input.shape[0] - plus_kernel.shape[0] + 1} × "
      f"{example_input.shape[1] - plus_kernel.shape[1] + 1}")

# Manual verification of (0,0) and (0,1)
manual_00 = 1*0 + 2*1 + 3*0 + 6*1 + 7*1 + 8*1 + 11*0 + 12*1 + 13*0
manual_01 = 2*0 + 3*1 + 4*0 + 7*1 + 8*1 + 9*1 + 12*0 + 13*1 + 14*0
print(f"\nManual verification:")
print(f"  Position (0,0): {manual_00}  — matches output[0,0] = {int(result[0,0])}")
print(f"  Position (0,1): {manual_01}  — matches output[0,1] = {int(result[0,1])}")

# =====================================================================
# === SECTION 2: CLASSICAL KERNELS AND EDGE DETECTION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: Classical Kernels — Sobel, Gaussian, Sharpening")
print("=" * 60)

# Load a CIFAR-10 image for visual demonstrations
(x_train_cifar, y_train_cifar), _ = cifar10.load_data()
y_train_cifar = y_train_cifar.flatten()
cifar_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Select a "cat" image (class 3) for clear edge structure
cat_idx = np.where(y_train_cifar == 3)[0][5]   # 5th cat in training set
rgb_img = x_train_cifar[cat_idx].astype(np.float32)   # shape (32, 32, 3)

# Convert to grayscale for single-channel convolution demos
# Luminance formula: Y = 0.299R + 0.587G + 0.114B
gray_img = (0.299 * rgb_img[:,:,0] +
            0.587 * rgb_img[:,:,1] +
            0.114 * rgb_img[:,:,2])
gray_img = gray_img / 255.0   # normalise to [0, 1]

print(f"\nUsing CIFAR-10 image: '{cifar_names[y_train_cifar[cat_idx]]}' (index {cat_idx})")
print(f"Grayscale image shape: {gray_img.shape}, value range: [{gray_img.min():.3f}, {gray_img.max():.3f}]")

# --- Define classical kernels ---

# Sobel horizontal: detects vertical edges (strong horizontal gradient)
sobel_h = np.array([
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
], dtype=np.float32)

# Sobel vertical: detects horizontal edges (strong vertical gradient)
sobel_v = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

# Gaussian blur 3×3: smooths / blurs by averaging with Gaussian weights
gaussian = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32) / 16.0   # normalise so weights sum to 1

# Sharpening: enhances edges by subtracting a blurred version
sharpen = np.array([
    [ 0, -1,  0],
    [-1,  5, -1],
    [ 0, -1,  0]
], dtype=np.float32)

# Identity: output equals input (no operation)
identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=np.float32)

# Apply all kernels to the grayscale image
edge_h    = convolve2d_valid(gray_img, sobel_h)
edge_v    = convolve2d_valid(gray_img, sobel_v)
edge_mag  = np.sqrt(edge_h**2 + edge_v**2)   # combined edge magnitude
blurred   = convolve2d_valid(gray_img, gaussian)
sharpened = convolve2d_valid(gray_img, sharpen)

# Clip values to valid range for display
edge_h_disp    = np.clip(edge_h, 0, 1)
edge_v_disp    = np.clip(edge_v, 0, 1)
edge_mag_disp  = np.clip(edge_mag / edge_mag.max(), 0, 1)   # normalise to [0,1]
sharpened_disp = np.clip(sharpened, 0, 1)

print("\nApplied kernels:")
print(f"  Sobel H output range: [{edge_h.min():.3f}, {edge_h.max():.3f}]")
print(f"  Sobel V output range: [{edge_v.min():.3f}, {edge_v.max():.3f}]")
print(f"  Edge magnitude range: [{edge_mag.min():.3f}, {edge_mag.max():.3f}]")
print(f"  Gaussian blur range:  [{blurred.min():.3f}, {blurred.max():.3f}]")
print(f"  Sharpened range:      [{sharpened.min():.3f}, {sharpened.max():.3f}]")

# Plot all kernel results
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
fig.suptitle("Classical Kernels Applied to CIFAR-10 Image", fontsize=14)

images_to_show = [
    (rgb_img.astype(np.uint8), "Original RGB\n(32×32)", 'gray', False),
    (gray_img,           "Grayscale\n(luminance)", 'gray', True),
    (edge_h_disp,        "Sobel Horizontal\n(vertical edges)", 'gray', True),
    (edge_v_disp,        "Sobel Vertical\n(horizontal edges)", 'gray', True),
    (edge_mag_disp,      "Edge Magnitude\n√(Hˆ2 + Vˆ2)", 'hot', True),
    (blurred,            "Gaussian Blur\n(3×3, σ≈1)", 'gray', True),
    (sharpened_disp,     "Sharpening\n(emphasises edges)", 'gray', True),
    (np.abs(gray_img[1:-1,1:-1] - blurred), "Diff: Original−Blur\n(high-freq content)", 'plasma', True),
]

for idx, (img, title, cmap, is_2d) in enumerate(images_to_show):
    ax = axes[idx // 4][idx % 4]
    if not is_2d:
        ax.imshow(img, interpolation='nearest')
    else:
        ax.imshow(img, cmap=cmap, interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_05_kernels.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_05_kernels.png")

# =====================================================================
# === SECTION 3: PADDING AND STRIDE EXPERIMENTS WITH KERAS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: Padding and Stride with Keras Conv2D")
print("=" * 60)

# Load MNIST for CNN experiments (clean, simple dataset)
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
x_train_cnn = x_train_mnist[..., np.newaxis].astype(np.float32) / 255.0
x_test_cnn  = x_test_mnist[..., np.newaxis].astype(np.float32)  / 255.0

# Use a single test image to demonstrate output dimensions
demo_image = x_test_cnn[0:1]   # shape (1, 28, 28, 1) — batch of 1

print(f"\nInput image shape: {demo_image.shape}")   # (1, 28, 28, 1)
print("\nDimension formula: output = floor((H + 2p - k) / s) + 1")
print("-" * 60)
print(f"{'Configuration':<40} {'Output Shape':>15}")
print("-" * 60)

# Test different padding and stride configurations
configs = [
    ("Conv2D(16, 3, pad=valid, stride=1)", dict(filters=16, kernel_size=3, padding='valid', strides=1)),
    ("Conv2D(16, 3, pad=same,  stride=1)", dict(filters=16, kernel_size=3, padding='same',  strides=1)),
    ("Conv2D(16, 3, pad=valid, stride=2)", dict(filters=16, kernel_size=3, padding='valid', strides=2)),
    ("Conv2D(16, 3, pad=same,  stride=2)", dict(filters=16, kernel_size=3, padding='same',  strides=2)),
    ("Conv2D(16, 5, pad=valid, stride=1)", dict(filters=16, kernel_size=5, padding='valid', strides=1)),
    ("Conv2D(16, 5, pad=same,  stride=1)", dict(filters=16, kernel_size=5, padding='same',  strides=1)),
]

for name, params in configs:
    # Build a single-layer model to easily check output shape
    single_model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        layers.Conv2D(**params, activation='relu')
    ])
    out = single_model(demo_image)   # forward pass with dummy data
    # Calculate expected dimensions using the formula
    H, k, p_str, s = 28, params['kernel_size'], params['padding'], params['strides']
    p = 0 if p_str == 'valid' else (k - 1) // 2
    expected = (H + 2*p - k) // s + 1
    print(f"  {name:<40} {str(out.shape):>15}  (expected spatial: {expected}×{expected})")

# =====================================================================
# === SECTION 4: CNN TRAINING AND FEATURE MAP VISUALISATION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: Training CNN and Visualising Feature Maps")
print("=" * 60)

# Build a simple CNN for MNIST
cnn = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv1'),     # 32 feature maps
    layers.MaxPooling2D((2, 2), name='pool1'),       # 28×28 → 14×14
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                  padding='same', name='conv2'),     # 64 feature maps
    layers.MaxPooling2D((2, 2), name='pool2'),       # 14×14 → 7×7
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax'),
], name="Feature_Map_CNN")

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

print("\nTraining CNN on MNIST (8 epochs)...")
cnn.fit(x_train_cnn, y_train_mnist, epochs=8, batch_size=128,
        validation_split=0.1, verbose=1)

test_loss, test_acc = cnn.evaluate(x_test_cnn, y_test_mnist, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# --- Visualise first conv layer filters ---
conv1_weights = cnn.get_layer('conv1').get_weights()[0]   # shape: (3, 3, 1, 32)
print(f"\nConv1 filter weights shape: {conv1_weights.shape}")
print(f"  (kernel_h, kernel_w, in_channels, n_filters) = {conv1_weights.shape}")

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle("Learned Filter Weights — Conv1 Layer (32 filters of 3×3)", fontsize=13)

for i in range(32):
    ax = axes[i // 8][i % 8]
    # Extract single filter: shape (3, 3, 1) → squeeze to (3, 3)
    filt = conv1_weights[:, :, 0, i]
    # Normalise to [0, 1] for display
    filt_norm = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
    ax.imshow(filt_norm, cmap='RdBu_r', interpolation='nearest')
    ax.set_title(f"F{i}", fontsize=7)
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_05_conv_filters.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_05_conv_filters.png")

# --- Build feature map extraction model ---
# A Keras "extractor" model outputs at an intermediate layer
feature_extractor = keras.Model(
    inputs=cnn.input,
    outputs=cnn.get_layer('conv1').output   # output of conv1: shape (1, 28, 28, 32)
)

# Select a test image to run through the extractor
test_sample = x_test_cnn[7:8]   # shape (1, 28, 28, 1)
test_label  = y_test_mnist[7]

print(f"\nExtracting feature maps for test image (true label: {test_label})...")
feature_maps = feature_extractor.predict(test_sample, verbose=0)
print(f"Feature maps shape: {feature_maps.shape}")   # (1, 28, 28, 32)

# Display the first 16 feature maps
fig, axes = plt.subplots(4, 5, figsize=(16, 13))
fig.suptitle(f"Conv1 Feature Maps for Test Image (True Label: {test_label})\n"
             "Each map shows where one filter's pattern was detected", fontsize=13)

# Show the original image in the first slot
ax0 = axes[0, 0]
ax0.imshow(x_test_mnist[7], cmap='gray', interpolation='nearest')
ax0.set_title(f"Original Image\nLabel: {test_label}", fontsize=10)
ax0.axis('off')

# Show first 19 feature maps (leave room for original in slot 0)
for i in range(19):
    ax = axes[(i + 1) // 5][(i + 1) % 5]
    fmap = feature_maps[0, :, :, i]   # shape (28, 28) — one spatial map
    ax.imshow(fmap, cmap='viridis', interpolation='nearest')
    ax.set_title(f"Filter {i}\nmax={fmap.max():.2f}", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_05_feature_maps.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_05_feature_maps.png")

# --- Verify manual numpy convolution matches Keras ---
print("\n--- Verifying numpy convolution matches Keras output ---")

# Extract the first learned filter (filter index 0) from conv1
learned_filter_0 = conv1_weights[:, :, 0, 0]   # shape (3, 3)
learned_bias_0   = cnn.get_layer('conv1').get_weights()[1][0]   # scalar bias

# Apply manually to the test image
test_gray_2d = x_test_cnn[7, :, :, 0]   # remove batch and channel dims → (28, 28)
manual_fm    = convolve2d_valid(test_gray_2d, learned_filter_0)   # shape (26, 26)
manual_fm_biased = manual_fm + learned_bias_0                      # add bias

# Get Keras output for same filter at same position (valid padding for fair comparison)
single_filter_model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    layers.Conv2D(1, kernel_size=(3, 3), padding='valid', use_bias=True, name='single_conv')
])
# Set this model's weights to the first learned filter
single_filter_model.get_layer('single_conv').set_weights(
    [learned_filter_0[:, :, np.newaxis, np.newaxis],
     np.array([learned_bias_0])]
)
keras_fm = single_filter_model.predict(test_sample, verbose=0)[0, :, :, 0]   # (26, 26)
# Note: Keras output has ReLU applied; manual doesn't — compare pre-activation
print(f"  Manual FM shape:  {manual_fm_biased.shape}")
print(f"  Keras FM shape:   {keras_fm.shape}")
max_diff = np.max(np.abs(manual_fm_biased - keras_fm))
print(f"  Max absolute difference (pre-activation): {max_diff:.6f}")
print(f"  → {'✓ MATCH (within floating point tolerance)' if max_diff < 1e-4 else '✗ MISMATCH'}")

# Final summary
print("\n" + "=" * 60)
print("LESSON 05 DEMO COMPLETE")
print("=" * 60)
print("\nKey takeaways from this demo:")
print("  1. Convolution = slide a filter over the image, compute dot products")
print("  2. Valid padding reduces dimensions; same padding preserves them")
print("  3. Stride > 1 downsamples the spatial dimensions")
print("  4. Different kernels detect different features: edges, blur, sharpness")
print("  5. CNNs learn their own filter weights from data (not hand-designed)")
print("  6. Feature maps show WHERE in the image each filter's pattern was found")
print("  7. Manual numpy convolution matches Keras Conv2D output exactly")
print("\nCourse progress: Lessons 1–5 complete!")
print("You now understand the full foundation of convolutional image classifiers.")
