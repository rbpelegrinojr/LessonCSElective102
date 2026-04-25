"""
Lesson 5: The Convolution Operation Deep Dive
Implements convolution from scratch and demonstrates filter effects.

Run: python code/lesson_05.py
Dependencies: tensorflow, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)

# =============================================================================
# SECTION 1: Manual Convolution Implementation from Scratch
# =============================================================================
print("=" * 60)
print("SECTION 1: Manual Convolution from Scratch (NumPy)")
print("=" * 60)
print("\nImplementing the convolution operation without any library.")
print("Output[i,j] = sum(Input[i:i+F, j:j+F] * Kernel)\n")

def convolve2d_manual(image, kernel, stride=1, padding=0):
    """
    Perform 2D cross-correlation (what CNNs call 'convolution').
    Args:
        image   : 2D numpy array (H, W)
        kernel  : 2D numpy array (F, F)
        stride  : int, step size for sliding the kernel
        padding : int, zero-padding amount on each side
    Returns:
        feature_map: 2D numpy array
    """
    if padding > 0:
        image = np.pad(image, padding, mode="constant", constant_values=0)

    H, W = image.shape
    F = kernel.shape[0]
    out_h = (H - F) // stride + 1
    out_w = (W - F) // stride + 1
    feature_map = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            row_start = i * stride
            col_start = j * stride
            patch = image[row_start:row_start + F, col_start:col_start + F]
            feature_map[i, j] = (patch * kernel).sum()

    return feature_map

def output_size(W, F, P, S):
    return int((W - F + 2 * P) / S) + 1

small_input = np.array([
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
    [7, 8, 9, 2, 3],
    [1, 2, 3, 0, 1],
    [4, 5, 6, 1, 2],
], dtype=np.float32)

identity_kernel = np.array([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 0]], dtype=np.float32)

edge_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype=np.float32)

print(f"Input shape  : {small_input.shape}")
print(f"Kernel size  : 3×3")
print(f"Stride=1, Padding=0:")
print(f"  Expected output size: {output_size(5, 3, 0, 1)} × {output_size(5, 3, 0, 1)}")
fm_identity = convolve2d_manual(small_input, identity_kernel, stride=1, padding=0)
fm_edge = convolve2d_manual(small_input, edge_kernel, stride=1, padding=0)
print(f"  Identity filter output:\n{fm_identity}")
print(f"  Edge filter output:\n{fm_edge}")

print(f"\nStride=2, Padding=0:")
print(f"  Expected output size: {output_size(5, 3, 0, 2)} × {output_size(5, 3, 0, 2)}")
fm_stride2 = convolve2d_manual(small_input, edge_kernel, stride=2, padding=0)
print(f"  Output:\n{fm_stride2}")

print(f"\nStride=1, Padding=1 ('same' for 3×3):")
print(f"  Expected output size: {output_size(5, 3, 1, 1)} × {output_size(5, 3, 1, 1)}")
fm_padded = convolve2d_manual(small_input, edge_kernel, stride=1, padding=1)
print(f"  Output:\n{fm_padded}")

from tensorflow.keras.datasets import mnist
try:
    (x_train, y_train), _ = mnist.load_data()
    print("MNIST loaded from Keras datasets.")
except Exception as e:
    print(f"Could not download MNIST ({e}). Using synthetic data.\n")
    rng_data = np.random.default_rng(0)
    x_train = rng_data.integers(0, 256, (60000, 28, 28), dtype=np.uint8)
    y_train = rng_data.integers(0, 10, 60000)
test_img_raw = x_train[0].astype(np.float32) / 255.0

fm_test = convolve2d_manual(test_img_raw, edge_kernel, stride=1, padding=1)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("SECTION 1: Manual Convolution from Scratch", fontsize=13, fontweight="bold")

axes[0].imshow(test_img_raw, cmap="gray")
axes[0].set_title(f"Original MNIST digit\n{test_img_raw.shape}", fontsize=10)
axes[0].axis("off")

axes[1].imshow(fm_test, cmap="hot")
axes[1].set_title(f"After edge kernel\n{fm_test.shape}", fontsize=10)
axes[1].axis("off")

axes[2].imshow(small_input, cmap="Blues")
for i in range(5):
    for j in range(5):
        axes[2].text(j, i, str(int(small_input[i, j])), ha="center", va="center",
                     fontsize=11, color="black")
axes[2].set_title("Small 5×5 input example\n(used in manual demo)", fontsize=10)
axes[2].axis("off")

plt.tight_layout()
plt.savefig("section1_manual_conv.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section1_manual_conv.png]")

# =============================================================================
# SECTION 2: Edge Detection with Sobel Filters
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: Edge Detection — Sobel Filters")
print("=" * 60)
print("\nSobel filters detect horizontal and vertical edges.")
print("They are fundamental to classical and learned image processing.\n")

sobel_x = np.array([[ 1,  0, -1],
                     [ 2,  0, -2],
                     [ 1,  0, -1]], dtype=np.float32)

sobel_y = np.array([[ 1,  2,  1],
                     [ 0,  0,  0],
                     [-1, -2, -1]], dtype=np.float32)

laplacian = np.array([[0,  1, 0],
                       [1, -4, 1],
                       [0,  1, 0]], dtype=np.float32)

sample_img = x_train[7].astype(np.float32) / 255.0

fm_sx = convolve2d_manual(sample_img, sobel_x, stride=1, padding=1)
fm_sy = convolve2d_manual(sample_img, sobel_y, stride=1, padding=1)
fm_magnitude = np.sqrt(fm_sx ** 2 + fm_sy ** 2)
fm_lap = convolve2d_manual(sample_img, laplacian, stride=1, padding=1)

print(f"Input image shape: {sample_img.shape}")
print(f"Sobel-X response  — min: {fm_sx.min():.3f}, max: {fm_sx.max():.3f}")
print(f"Sobel-Y response  — min: {fm_sy.min():.3f}, max: {fm_sy.max():.3f}")
print(f"Gradient magnitude— min: {fm_magnitude.min():.3f}, max: {fm_magnitude.max():.3f}")
print(f"\nSobel-X kernel (detects vertical edges — left-to-right transitions):")
print(sobel_x.astype(int))
print(f"\nSobel-Y kernel (detects horizontal edges — top-to-bottom transitions):")
print(sobel_y.astype(int))

fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle("SECTION 2: Sobel Edge Detection Filters", fontsize=13, fontweight="bold")

plot_configs = [
    (sample_img, "gray", f"Original\n(digit {y_train[7]})"),
    (fm_sx, "RdBu", "Sobel-X\n(vertical edges)"),
    (fm_sy, "RdBu", "Sobel-Y\n(horizontal edges)"),
    (fm_magnitude, "hot", "Gradient Magnitude\nsqrt(Sx²+Sy²)"),
    (fm_lap, "RdBu", "Laplacian\n(all edges)"),
]
for ax, (data, cmap, title) in zip(axes, plot_configs):
    ax.imshow(data, cmap=cmap)
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("section2_edge_detection.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section2_edge_detection.png]")

# =============================================================================
# SECTION 3: Blur and Sharpen Filters — Side-by-Side Comparison
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Blur and Sharpen Filters")
print("=" * 60)
print("\nDifferent kernels transform the image in fundamentally different ways.")
print("Understanding these helps interpret what CNN filters learn.\n")

box_blur = np.ones((3, 3), dtype=np.float32) / 9.0

gaussian_blur = np.array([[1, 2, 1],
                            [2, 4, 2],
                            [1, 2, 1]], dtype=np.float32) / 16.0

sharpen = np.array([[ 0, -1,  0],
                     [-1,  5, -1],
                     [ 0, -1,  0]], dtype=np.float32)

emboss = np.array([[-2, -1,  0],
                    [-1,  1,  1],
                    [ 0,  1,  2]], dtype=np.float32)

original = x_train[5].astype(np.float32) / 255.0
fm_box = np.clip(convolve2d_manual(original, box_blur, padding=1), 0, 1)
fm_gauss = np.clip(convolve2d_manual(original, gaussian_blur, padding=1), 0, 1)
fm_sharp = np.clip(convolve2d_manual(original, sharpen, padding=1), 0, 1)
fm_emboss = convolve2d_manual(original, emboss, padding=1)
fm_emboss = (fm_emboss - fm_emboss.min()) / (fm_emboss.max() - fm_emboss.min() + 1e-9)

filters_display = [
    ("Box Blur\n(average of 3×3)", box_blur),
    ("Gaussian Blur\n(weighted average)", gaussian_blur),
    ("Sharpen\n(enhances edges)", sharpen),
    ("Emboss\n(3D illusion)", emboss),
]

print("Filter kernels:")
for name, kernel in filters_display:
    print(f"\n  {name.replace(chr(10), ' — ')}:")
    for row in kernel:
        print("    " + "  ".join(f"{v:5.2f}" for v in row))

fig, axes = plt.subplots(2, 5, figsize=(18, 7))
fig.suptitle("SECTION 3: Blur, Sharpen, and Emboss Filter Comparison",
             fontsize=13, fontweight="bold")

results = [original, fm_box, fm_gauss, fm_sharp, fm_emboss]
titles_top = [f"Original\n(digit {y_train[5]})", "Box Blur", "Gaussian Blur", "Sharpen", "Emboss"]

for col, (img, title) in enumerate(zip(results, titles_top)):
    axes[0, col].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, col].set_title(title, fontsize=10)
    axes[0, col].axis("off")

for col, (name, kernel) in enumerate(filters_display):
    im = axes[1, col + 1].imshow(kernel, cmap="RdBu_r",
                                  vmin=-kernel.max(), vmax=kernel.max())
    axes[1, col + 1].set_title(f"Kernel:\n{name.split(chr(10))[0]}", fontsize=9)
    for r in range(kernel.shape[0]):
        for c in range(kernel.shape[1]):
            axes[1, col + 1].text(c, r, f"{kernel[r, c]:.2f}",
                                   ha="center", va="center", fontsize=8)
    axes[1, col + 1].axis("off")

axes[1, 0].text(0.5, 0.5, "Filter kernels\nvisualized below\n(blue=negative\nred=positive)",
                ha="center", va="center", fontsize=10, transform=axes[1, 0].transAxes)
axes[1, 0].axis("off")

plt.tight_layout()
plt.savefig("section3_filters.png", dpi=100, bbox_inches="tight")
print("[Saved: section3_filters.png]")

# =============================================================================
# SECTION 4: Keras Conv2D Layer — Examining and Setting Filters
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Keras Conv2D — Examining and Setting Filter Weights")
print("=" * 60)
print("\nIn practice we use Keras Conv2D. Let's examine how it stores filters")
print("and demonstrate that we can read and write the learned weights.\n")

conv_layer = tf.keras.layers.Conv2D(
    filters=4,
    kernel_size=(3, 3),
    padding="same",
    activation="relu",
    name="demo_conv",
)
dummy_input = tf.zeros((1, 28, 28, 1))
_ = conv_layer(dummy_input)

kernel_weights, bias_weights = conv_layer.get_weights()
print(f"Kernel weights shape : {kernel_weights.shape}")
print(f"  → (kernel_h, kernel_w, in_channels, out_filters) = (3, 3, 1, 4)")
print(f"Bias weights shape   : {bias_weights.shape}  → one bias per filter")
print(f"\nTotal parameters     : {kernel_weights.size + bias_weights.size}")
print(f"  Breakdown: (3×3×1×4) + 4 = {kernel_weights.size} + {bias_weights.size}")

print(f"\nRandom initialized kernel[..., 0, 0] (first filter, first input channel):")
print(np.round(kernel_weights[:, :, 0, 0], 4))

custom_kernels = np.zeros((3, 3, 1, 4), dtype=np.float32)
custom_kernels[:, :, 0, 0] = sobel_x
custom_kernels[:, :, 0, 1] = sobel_y
custom_kernels[:, :, 0, 2] = gaussian_blur
custom_kernels[:, :, 0, 3] = sharpen
custom_biases = np.zeros(4, dtype=np.float32)

conv_layer.set_weights([custom_kernels, custom_biases])
print("\nSet custom kernels: [Sobel-X, Sobel-Y, Gaussian-Blur, Sharpen]")

test_input = x_train[0].astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
output_maps = conv_layer(test_input).numpy()[0]
print(f"\nInput shape  : {test_input.shape}")
print(f"Output shape : {output_maps.shape}  → (H, W, num_filters)")
print(f"  ✓ Same H,W because padding='same'; 4 feature maps (one per filter)")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("SECTION 4: Keras Conv2D — Custom Filters and Their Outputs",
             fontsize=13, fontweight="bold")

kernel_names = ["Sobel-X", "Sobel-Y", "Gaussian Blur", "Sharpen"]
for col in range(4):
    k = custom_kernels[:, :, 0, col]
    vmax = np.abs(k).max()
    axes[0, col].imshow(k, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0, col].set_title(f"Kernel: {kernel_names[col]}", fontsize=10)
    for r in range(3):
        for c in range(3):
            axes[0, col].text(c, r, f"{k[r, c]:.2f}",
                               ha="center", va="center", fontsize=9,
                               color="white" if abs(k[r, c]) > vmax * 0.5 else "black")
    axes[0, col].axis("off")

    axes[1, col].imshow(output_maps[:, :, col], cmap="hot")
    axes[1, col].set_title(f"Feature Map: {kernel_names[col]}", fontsize=10)
    axes[1, col].axis("off")

plt.tight_layout()
plt.savefig("section4_keras_conv.png", dpi=100, bbox_inches="tight")
print("[Saved: section4_keras_conv.png]")

# =============================================================================
# SECTION 5: Feature Map Visualization — Multiple Filters
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Feature Map Visualization — Multiple Filters on Real Image")
print("=" * 60)
print("\nApplying many filters shows how different filters detect different features.")
print("This is what the first convolutional layer of a CNN does.\n")

multi_conv = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv1"),
], name="feature_extractor")

multi_conv.summary()

input_tensor = x_train[0].astype(np.float32)[np.newaxis, :, :, np.newaxis] / 255.0
all_feature_maps = multi_conv.predict(input_tensor, verbose=0)[0]
print(f"\nInput shape         : {input_tensor.shape}")
print(f"All feature maps    : {all_feature_maps.shape}")
print(f"  → {all_feature_maps.shape[2]} filters, each producing a {all_feature_maps.shape[:2]} map")

print("\nFeature map statistics per filter:")
print(f"  {'Filter':<8} {'Min':>8} {'Max':>8} {'Mean':>8} {'Std':>8}")
print("  " + "-" * 40)
for i in range(all_feature_maps.shape[2]):
    fm = all_feature_maps[:, :, i]
    print(f"  {i:<8} {fm.min():>8.3f} {fm.max():>8.3f} {fm.mean():>8.3f} {fm.std():>8.3f}")

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f"SECTION 5: All 16 Feature Maps from First Conv Layer\n"
             f"(Random initialization — each filter detects different low-level patterns)",
             fontsize=12, fontweight="bold")

ax_orig = fig.add_subplot(4, 5, 1)
ax_orig.imshow(x_train[0], cmap="gray")
ax_orig.set_title(f"Original\n(digit {y_train[0]})", fontsize=10, fontweight="bold",
                  color="darkred")
ax_orig.axis("off")

for i in range(16):
    ax = fig.add_subplot(4, 5, i + 2)
    fm = all_feature_maps[:, :, i]
    vmax = np.abs(fm).max()
    ax.imshow(fm, cmap="viridis")
    ax.set_title(f"Filter {i}\nmax={fm.max():.2f}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("section5_feature_maps.png", dpi=100, bbox_inches="tight")
print("[Saved: section5_feature_maps.png]")

print("\nNow showing feature maps from a TRAINED model (better filters):")

from tensorflow.keras.datasets import mnist as mnist_data
try:
    (x_tr, y_tr), (x_te, y_te) = mnist_data.load_data()
except Exception as e:
    print(f"Could not download MNIST ({e}). Using synthetic data.\n")
    rng_data = np.random.default_rng(1)
    x_tr = rng_data.integers(0, 256, (60000, 28, 28), dtype=np.uint8)
    y_tr = rng_data.integers(0, 10, 60000)
    x_te = rng_data.integers(0, 256, (10000, 28, 28), dtype=np.uint8)
    y_te = rng_data.integers(0, 10, 10000)
x_tr_n = x_tr.astype(np.float32)[..., np.newaxis] / 255.0
x_te_n = x_te.astype(np.float32)[..., np.newaxis] / 255.0

trained_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", padding="same", name="conv_trained"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
], name="trained_extractor")

trained_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
trained_model.fit(x_tr_n, y_tr, epochs=3, batch_size=256,
                  validation_data=(x_te_n, y_te), verbose=1)

extractor = tf.keras.Model(
    inputs=trained_model.layers[0].input,
    outputs=trained_model.get_layer("conv_trained").output,
)
trained_maps = extractor.predict(input_tensor, verbose=0)[0]

fig2 = plt.figure(figsize=(18, 10))
fig2.suptitle("SECTION 5: Feature Maps from TRAINED Conv Layer (3 epochs)\n"
              "(Compare to random — trained filters detect structured patterns!)",
              fontsize=12, fontweight="bold")

ax_orig2 = fig2.add_subplot(4, 5, 1)
ax_orig2.imshow(x_train[0], cmap="gray")
ax_orig2.set_title(f"Original\n(digit {y_train[0]})", fontsize=10, fontweight="bold",
                   color="darkred")
ax_orig2.axis("off")

for i in range(16):
    ax = fig2.add_subplot(4, 5, i + 2)
    fm = trained_maps[:, :, i]
    ax.imshow(fm, cmap="viridis")
    ax.set_title(f"Filter {i}\nmax={fm.max():.2f}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.savefig("section5_trained_feature_maps.png", dpi=100, bbox_inches="tight")
print("[Saved: section5_trained_feature_maps.png]")

print("\n" + "=" * 60)
print("All 5 sections complete!")
print("Generated files:")
for fname in ["section1_manual_conv.png", "section2_edge_detection.png",
              "section3_filters.png", "section4_keras_conv.png",
              "section5_feature_maps.png", "section5_trained_feature_maps.png"]:
    print(f"  {fname}")
print("=" * 60)

plt.show()
