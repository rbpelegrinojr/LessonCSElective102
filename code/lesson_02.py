"""
Lesson 2: Understanding Digital Images & Pixels
Demonstrates how digital images are stored and manipulated as numerical arrays.

Run: python code/lesson_02.py
Dependencies: tensorflow, numpy, matplotlib, Pillow
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

# =============================================================================
# SECTION 1: Loading Images and Inspecting Dimensions / Dtypes
# =============================================================================
print("=" * 60)
print("SECTION 1: Loading Images and Inspecting Properties")
print("=" * 60)
print("\nWe will create synthetic images and also load real data from MNIST")
print("to explore pixel dimensions, data types, and array shapes.\n")

from tensorflow.keras.datasets import mnist

try:
    (x_train, _), _ = mnist.load_data()
    print("MNIST loaded from Keras datasets.")
except Exception as e:
    print(f"Could not download MNIST ({e}).")
    print("Generating synthetic MNIST-like data for demonstration.\n")
    rng_data = np.random.default_rng(0)
    x_train = rng_data.integers(0, 256, (60000, 28, 28), dtype=np.uint8)

mnist_img = x_train[0]
print("--- MNIST Grayscale Image ---")
print(f"  Shape  : {mnist_img.shape}  → (height, width) = 28×28 pixels")
print(f"  Dtype  : {mnist_img.dtype}  → uint8: unsigned 8-bit integer")
print(f"  Min px : {mnist_img.min()}")
print(f"  Max px : {mnist_img.max()}")
print(f"  Total pixels: {mnist_img.size}")

synthetic_color = np.zeros((64, 64, 3), dtype=np.uint8)
synthetic_color[:32, :32] = [255, 0, 0]
synthetic_color[:32, 32:] = [0, 255, 0]
synthetic_color[32:, :32] = [0, 0, 255]
synthetic_color[32:, 32:] = [255, 255, 0]

print("\n--- Synthetic Color Image (4 colored quadrants) ---")
print(f"  Shape  : {synthetic_color.shape}  → (height, width, channels)")
print(f"  Dtype  : {synthetic_color.dtype}")
print(f"  Memory : {synthetic_color.nbytes} bytes = {synthetic_color.nbytes / 1024:.2f} KB")

pil_image = Image.fromarray(synthetic_color)
print(f"\n  PIL Image size   : {pil_image.size}  → (width, height) — NOTE: PIL flips H,W order!")
print(f"  PIL Image mode   : {pil_image.mode}  → RGB")
print(f"  PIL Image format : {pil_image.format}  → None (created from array, not file)")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle("SECTION 1: Image Loading — Dimensions and Dtypes",
             fontsize=13, fontweight="bold")

axes[0].imshow(mnist_img, cmap="gray")
axes[0].set_title(f"MNIST digit\nShape: {mnist_img.shape}, dtype: {mnist_img.dtype}", fontsize=10)
axes[0].axis("off")

axes[1].imshow(synthetic_color)
axes[1].set_title(f"Synthetic RGB image\nShape: {synthetic_color.shape}, dtype: {synthetic_color.dtype}",
                  fontsize=10)
axes[1].axis("off")

axes[2].text(0.1, 0.8, "Image Terminology:", fontsize=12, fontweight="bold",
             transform=axes[2].transAxes)
lines = [
    "Shape (H, W)    = grayscale",
    "Shape (H, W, 1) = grayscale (explicit)",
    "Shape (H, W, 3) = RGB color",
    "Shape (H, W, 4) = RGBA (with transparency)",
    "",
    "uint8  → values 0–255 (on disk)",
    "float32 → values 0.0–1.0 (for NN)",
]
for i, line in enumerate(lines):
    axes[2].text(0.05, 0.65 - i * 0.08, line, fontsize=9, transform=axes[2].transAxes,
                 family="monospace")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("section1_loading.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section1_loading.png]")

# =============================================================================
# SECTION 2: RGB Channels — Split and Visualize
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: RGB Channels — Splitting and Visualizing")
print("=" * 60)
print("\nAn RGB image is three stacked 2D arrays.")
print("Splitting channels shows what each color component contributes.\n")

gradient_r = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (256, 1))
gradient_g = np.tile(np.linspace(0, 200, 256, dtype=np.uint8).reshape(-1, 1), (1, 256))
gradient_b = np.full((256, 256), 128, dtype=np.uint8)

rgb_demo = np.stack([gradient_r, gradient_g, gradient_b], axis=2)
print(f"Demo RGB image shape: {rgb_demo.shape}")

r_channel = rgb_demo[:, :, 0]
g_channel = rgb_demo[:, :, 1]
b_channel = rgb_demo[:, :, 2]

print(f"Red channel   shape: {r_channel.shape}, range: {r_channel.min()}–{r_channel.max()}")
print(f"Green channel shape: {g_channel.shape}, range: {g_channel.min()}–{g_channel.max()}")
print(f"Blue channel  shape: {b_channel.shape}, range: {b_channel.min()}–{b_channel.max()}")

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("SECTION 2: RGB Channel Decomposition", fontsize=13, fontweight="bold")

axes[0, 0].imshow(rgb_demo)
axes[0, 0].set_title("Original RGB Image", fontsize=11, fontweight="bold")
axes[0, 0].axis("off")

channel_configs = [
    (r_channel, "Reds", "Red Channel (R)"),
    (g_channel, "Greens", "Green Channel (G)"),
    (b_channel, "Blues", "Blue Channel (B)"),
]
for j, (channel, cmap, title) in enumerate(channel_configs):
    axes[0, j + 1].imshow(channel, cmap=cmap)
    axes[0, j + 1].set_title(title, fontsize=11)
    axes[0, j + 1].axis("off")

# Show each channel as grayscale intensity map
axes[1, 0].text(0.5, 0.5,
    "Bottom row shows\nthe same channels\nas grayscale\nintensity maps\n\n"
    "Bright = high value\nDark = low value",
    ha="center", va="center", fontsize=11, transform=axes[1, 0].transAxes)
axes[1, 0].axis("off")

for j, (channel, _, title) in enumerate(channel_configs):
    im = axes[1, j + 1].imshow(channel, cmap="gray", vmin=0, vmax=255)
    axes[1, j + 1].set_title(f"{title}\n(Grayscale intensity)", fontsize=10)
    axes[1, j + 1].axis("off")
    plt.colorbar(im, ax=axes[1, j + 1], fraction=0.046)

plt.tight_layout()
plt.savefig("section2_rgb_channels.png", dpi=100, bbox_inches="tight")
print("[Saved: section2_rgb_channels.png]")

# =============================================================================
# SECTION 3: Grayscale Conversion and Comparison
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Grayscale Conversion and Comparison")
print("=" * 60)
print("\nConverting RGB to grayscale discards color information")
print("but reduces data by 2/3 and simplifies processing.\n")

print("Luminosity formula: Gray = 0.299×R + 0.587×G + 0.114×B")
print("Weights reflect human eye sensitivity: G > R > B\n")

r = rgb_demo[:, :, 0].astype(np.float32)
g = rgb_demo[:, :, 1].astype(np.float32)
b = rgb_demo[:, :, 2].astype(np.float32)

grayscale_luminosity = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
grayscale_average = ((r + g + b) / 3).astype(np.uint8)
grayscale_max = np.maximum(np.maximum(r, g), b).astype(np.uint8)

print(f"Original RGB     : shape={rgb_demo.shape}, size={rgb_demo.nbytes} bytes")
print(f"Grayscale result : shape={grayscale_luminosity.shape}, size={grayscale_luminosity.nbytes} bytes")
print(f"Memory saved     : {(1 - grayscale_luminosity.nbytes/rgb_demo.nbytes)*100:.1f}%\n")

pil_original = Image.fromarray(rgb_demo)
pil_gray = pil_original.convert("L")
grayscale_pil = np.array(pil_gray)
print(f"PIL grayscale shape: {grayscale_pil.shape}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("SECTION 3: Grayscale Conversion Methods", fontsize=13, fontweight="bold")

axes[0].imshow(rgb_demo)
axes[0].set_title("Original RGB", fontsize=11)
axes[0].axis("off")

methods = [
    (grayscale_luminosity, "Luminosity\n0.299R+0.587G+0.114B"),
    (grayscale_average, "Simple Average\n(R+G+B)/3"),
    (grayscale_max, "Max Channel\nmax(R,G,B)"),
]
for ax, (gray, title) in zip(axes[1:], methods):
    ax.imshow(gray, cmap="gray")
    ax.set_title(title, fontsize=10)
    ax.axis("off")

plt.tight_layout()
plt.savefig("section3_grayscale.png", dpi=100, bbox_inches="tight")
print("[Saved: section3_grayscale.png]")

# =============================================================================
# SECTION 4: Normalization — From 0–255 to 0.0–1.0
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Normalization — Scaling Pixel Values for Neural Networks")
print("=" * 60)
print("\nNeural networks expect float inputs, ideally in [0, 1] or [-1, 1].")
print("Normalization stabilizes gradient descent and speeds up training.\n")

raw_image = x_train[:100]
print(f"Raw uint8 image:")
print(f"  dtype  : {raw_image.dtype}")
print(f"  min    : {raw_image.min()}")
print(f"  max    : {raw_image.max()}")
print(f"  mean   : {raw_image.mean():.2f}")
print(f"  std    : {raw_image.std():.2f}")

normalized_01 = raw_image.astype(np.float32) / 255.0
print(f"\nAfter dividing by 255 (min-max normalization):")
print(f"  dtype  : {normalized_01.dtype}")
print(f"  min    : {normalized_01.min():.4f}")
print(f"  max    : {normalized_01.max():.4f}")
print(f"  mean   : {normalized_01.mean():.4f}")
print(f"  std    : {normalized_01.std():.4f}")

dataset_mean = normalized_01.mean()
dataset_std = normalized_01.std()
standardized = (normalized_01 - dataset_mean) / dataset_std
print(f"\nAfter z-score standardization (mean=0, std=1):")
print(f"  dtype  : {standardized.dtype}")
print(f"  min    : {standardized.min():.4f}")
print(f"  max    : {standardized.max():.4f}")
print(f"  mean   : {standardized.mean():.6f}  ← ≈ 0")
print(f"  std    : {standardized.std():.6f}   ← ≈ 1")

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("SECTION 4: Normalization Effects", fontsize=13, fontweight="bold")

sample = x_train[0]

axes[0, 0].imshow(sample, cmap="gray")
axes[0, 0].set_title(f"Raw uint8\nmin=0, max=255", fontsize=10)
axes[0, 0].axis("off")

axes[0, 1].imshow(sample.astype(np.float32) / 255.0, cmap="gray")
axes[0, 1].set_title("Normalized float32\nmin=0.0, max=1.0", fontsize=10)
axes[0, 1].axis("off")

axes[0, 2].imshow((standardized[0] - standardized[0].min()) /
                  (standardized[0].max() - standardized[0].min()), cmap="gray")
axes[0, 2].set_title("Standardized\n(rescaled to [0,1] for display)", fontsize=10)
axes[0, 2].axis("off")

axes[1, 0].hist(raw_image.flatten(), bins=50, color="tomato", edgecolor="black")
axes[1, 0].set_title("Pixel Distribution (uint8)\n0–255", fontsize=10)
axes[1, 0].set_xlabel("Pixel Value")
axes[1, 0].set_ylabel("Count")

axes[1, 1].hist(normalized_01.flatten(), bins=50, color="steelblue", edgecolor="black")
axes[1, 1].set_title("Pixel Distribution (float32)\n0.0–1.0", fontsize=10)
axes[1, 1].set_xlabel("Pixel Value")

axes[1, 2].hist(standardized.flatten(), bins=50, color="seagreen", edgecolor="black")
axes[1, 2].set_title("Pixel Distribution (standardized)\nmean≈0, std≈1", fontsize=10)
axes[1, 2].set_xlabel("Pixel Value")

plt.tight_layout()
plt.savefig("section4_normalization.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section4_normalization.png]")

# =============================================================================
# SECTION 5: Batch Representation — Tensor Shape (N, H, W, C)
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Batch Representation — Creating Image Batches")
print("=" * 60)
print("\nNeural networks process images in batches for GPU efficiency.")
print("A batch is a 4D tensor: (batch_size, height, width, channels)\n")

batch_size = 8
single_mnist = x_train[:batch_size].astype(np.float32) / 255.0
single_mnist_with_channel = single_mnist[:, :, :, np.newaxis]

print(f"Single MNIST image shape        : {x_train[0].shape}")
print(f"  → (height, width)")

print(f"\nBatch of {batch_size} MNIST images (no channel dim) : {single_mnist.shape}")
print(f"  → (batch_size, height, width)")

print(f"\nBatch with explicit channel dim : {single_mnist_with_channel.shape}")
print(f"  → (batch_size, height, width, channels)  ← TensorFlow/Keras format")

print(f"\nMemory for this batch           : {single_mnist_with_channel.nbytes} bytes "
      f"= {single_mnist_with_channel.nbytes / 1024:.1f} KB")

batch_32_color = np.random.rand(32, 224, 224, 3).astype(np.float32)
memory_mb = batch_32_color.nbytes / (1024 ** 2)
print(f"\nTypical training batch (32 × 224×224 color float32):")
print(f"  Shape  : {batch_32_color.shape}")
print(f"  Memory : {memory_mb:.1f} MB")

print("\n--- Common tensor shapes in deep learning ---")
shapes = [
    ("Single grayscale MNIST", (28, 28), "No batch or channel dim"),
    ("Single MNIST for Keras", (1, 28, 28, 1), "Batch=1, channel=1"),
    ("Batch of 32 MNIST",  (32, 28, 28, 1), "Batch=32, channel=1 (grayscale)"),
    ("Batch of 16 CIFAR-10", (16, 32, 32, 3), "Batch=16, channels=3 (RGB)"),
    ("Batch of 8 ImageNet",  (8, 224, 224, 3), "Batch=8, channels=3 (RGB)"),
]
for name, shape, note in shapes:
    print(f"  {name:<28}: {str(shape):<22} {note}")

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
fig.suptitle("SECTION 5: Batch of 8 MNIST Images — Shape (8, 28, 28, 1)",
             fontsize=13, fontweight="bold")

for i in range(8):
    row, col = divmod(i, 4)
    ax = axes[row, col]
    ax.imshow(single_mnist_with_channel[i, :, :, 0], cmap="gray")
    ax.set_title(f"Batch[{i}]\nshape: {single_mnist_with_channel[i].shape}", fontsize=9)
    ax.axis("off")

plt.tight_layout()
plt.savefig("section5_batch.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section5_batch.png]")

print("\n" + "=" * 60)
print("All 5 sections complete!")
print("Generated files:")
for fname in ["section1_loading.png", "section2_rgb_channels.png",
              "section3_grayscale.png", "section4_normalization.png",
              "section5_batch.png"]:
    print(f"  {fname}")
print("=" * 60)

plt.show()
