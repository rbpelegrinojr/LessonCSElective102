"""
Lesson 02: Understanding Digital Images and Pixels
====================================================
This module demonstrates how digital images are stored as multi-dimensional
numpy arrays, explores grayscale vs RGB representations, shows how to
manipulate pixel values, convert between colour spaces, separate channels,
normalise pixel intensities, and plot histograms to analyse image statistics.

Datasets used:
  - MNIST: 28×28 grayscale images of handwritten digits
  - CIFAR-10: 32×32 RGB images across 10 object categories
"""

# === STANDARD LIBRARY & THIRD-PARTY IMPORTS ===
import numpy as np                              # array operations and math
import matplotlib.pyplot as plt                 # plotting library
import matplotlib.colors as mcolors            # colour utilities
from tensorflow.keras.datasets import mnist    # grayscale digit dataset
from tensorflow.keras.datasets import cifar10  # colour image dataset

# =====================================================================
# === SECTION 1: GRAYSCALE IMAGES — MNIST ===
# =====================================================================

print("=" * 60)
print("SECTION 1: Grayscale Images (MNIST)")
print("=" * 60)

# Load MNIST: images are uint8 numpy arrays of shape (N, 28, 28)
(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()

# Inspect the array properties of a single grayscale image
gray_img = x_train_mnist[0]   # shape: (28, 28), dtype: uint8

print(f"\nGrayscale image shape:  {gray_img.shape}")   # (28, 28)
print(f"Number of dimensions:   {gray_img.ndim}")      # 2
print(f"Data type:              {gray_img.dtype}")     # uint8
print(f"Min pixel value:        {gray_img.min()}")     # likely 0 (background)
print(f"Max pixel value:        {gray_img.max()}")     # likely 255 (ink)
print(f"Mean pixel value:       {gray_img.mean():.2f}")
print(f"Memory usage:           {gray_img.nbytes} bytes ({gray_img.nbytes / 1024:.2f} KB)")

# Demonstrate that we can access individual pixels by row and column index
print(f"\nPixel at (0, 0) top-left corner:    {gray_img[0, 0]}")    # background = 0
print(f"Pixel at (14, 14) centre of image:  {gray_img[14, 14]}")   # may be ink

# Show a small 8×8 patch from the centre of the image
patch = gray_img[10:18, 10:18]  # rows 10–17, cols 10–17
print(f"\n8×8 pixel patch from centre (rows 10–17, cols 10–17):")
print(patch)

# =====================================================================
# === SECTION 2: RGB COLOUR IMAGES — CIFAR-10 ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: RGB Colour Images (CIFAR-10)")
print("=" * 60)

# CIFAR-10 class names (in order, matching the integer labels 0–9)
cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                     'dog', 'frog', 'horse', 'ship', 'truck']

# Load CIFAR-10: images are uint8 numpy arrays of shape (N, 32, 32, 3)
(x_train_cifar, y_train_cifar), (x_test_cifar, y_test_cifar) = cifar10.load_data()

# Flatten the label arrays: Keras returns shape (N, 1) for CIFAR-10
y_train_cifar = y_train_cifar.flatten()   # shape (50000,)
y_test_cifar  = y_test_cifar.flatten()    # shape (10000,)

# Inspect the array properties of a single colour image
colour_img = x_train_cifar[0]   # shape: (32, 32, 3)
colour_label = y_train_cifar[0]

print(f"\nColour image shape:     {colour_img.shape}")   # (32, 32, 3)
print(f"Number of dimensions:   {colour_img.ndim}")      # 3
print(f"Data type:              {colour_img.dtype}")     # uint8
print(f"Image class:            {cifar_class_names[colour_label]} (label {colour_label})")
print(f"Total pixel values:     {colour_img.size}")      # 32*32*3 = 3072
print(f"Memory per image:       {colour_img.nbytes} bytes")

# Access individual channel values for a specific pixel
row, col = 16, 16  # centre pixel
r_val = colour_img[row, col, 0]   # Red channel value at centre
g_val = colour_img[row, col, 1]   # Green channel value at centre
b_val = colour_img[row, col, 2]   # Blue channel value at centre
print(f"\nCentre pixel (row={row}, col={col}) RGB values: "
      f"R={r_val}, G={g_val}, B={b_val}")

# =====================================================================
# === SECTION 3: CHANNEL SEPARATION AND VISUALISATION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: Separating RGB Channels")
print("=" * 60)

# Select a visually interesting CIFAR-10 image — find first "frog" (class 6)
frog_idx = np.where(y_train_cifar == 6)[0][0]
frog_img = x_train_cifar[frog_idx].copy()   # copy so we don't modify the dataset

print(f"\nSelected image: class 'frog' (index {frog_idx})")
print(f"Image shape: {frog_img.shape}")

# Extract individual colour channels
red_channel   = frog_img[:, :, 0]   # shape (32, 32) — Red only
green_channel = frog_img[:, :, 1]   # shape (32, 32) — Green only
blue_channel  = frog_img[:, :, 2]   # shape (32, 32) — Blue only

print(f"\nRed channel   — min: {red_channel.min()}, max: {red_channel.max()}, "
      f"mean: {red_channel.mean():.1f}")
print(f"Green channel — min: {green_channel.min()}, max: {green_channel.max()}, "
      f"mean: {green_channel.mean():.1f}")
print(f"Blue channel  — min: {blue_channel.min()}, max: {blue_channel.max()}, "
      f"mean: {blue_channel.mean():.1f}")
print("\nFrogs are green, so we expect the Green channel mean to be highest.")

# Create images where only one channel is active (for visualisation)
# Set the other two channels to zero to show the pure colour contribution
red_only   = np.zeros_like(frog_img);  red_only[:, :, 0]   = red_channel
green_only = np.zeros_like(frog_img);  green_only[:, :, 1] = green_channel
blue_only  = np.zeros_like(frog_img);  blue_only[:, :, 2]  = blue_channel

# Plot: original + 3 channel separations + 3 grayscale channels
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
fig.suptitle("RGB Channel Separation — CIFAR-10 Frog", fontsize=14)

# Row 0: colour representations
axes[0, 0].imshow(frog_img);        axes[0, 0].set_title("Original (RGB)")
axes[0, 1].imshow(red_only);        axes[0, 1].set_title("Red Channel Only")
axes[0, 2].imshow(green_only);      axes[0, 2].set_title("Green Channel Only")
axes[0, 3].imshow(blue_only);       axes[0, 3].set_title("Blue Channel Only")

# Row 1: grayscale representations of each channel (shows intensity distribution)
axes[1, 0].axis('off')   # placeholder
axes[1, 1].imshow(red_channel, cmap='Reds');     axes[1, 1].set_title("Red Intensity Map")
axes[1, 2].imshow(green_channel, cmap='Greens'); axes[1, 2].set_title("Green Intensity Map")
axes[1, 3].imshow(blue_channel, cmap='Blues');   axes[1, 3].set_title("Blue Intensity Map")

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_02_channel_separation.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_02_channel_separation.png")

# =====================================================================
# === SECTION 4: NORMALISATION AND COLOUR SPACE CONVERSION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: Normalisation and Grayscale Conversion")
print("=" * 60)

# --- Normalisation: dividing by 255 ---
print("\nNormalising pixel values from [0, 255] to [0.0, 1.0]...")

# Convert to float32 BEFORE dividing — integer division would give 0 or 1
frog_float = frog_img.astype(np.float32)         # copy with float dtype
frog_norm  = frog_float / 255.0                  # normalise to [0.0, 1.0]

print(f"  Raw image dtype:        {frog_img.dtype}")
print(f"  Raw image value range:  [{frog_img.min()}, {frog_img.max()}]")
print(f"  Normalised dtype:       {frog_norm.dtype}")
print(f"  Normalised value range: [{frog_norm.min():.4f}, {frog_norm.max():.4f}]")
print(f"  Normalised mean:        {frog_norm.mean():.4f}")

# Verify that specific pixel is correctly transformed
print(f"\n  Original pixel [0,0,0] = {frog_img[0,0,0]}")
print(f"  Normalised pixel [0,0,0] = {frog_norm[0,0,0]:.6f}")
print(f"  Expected: {frog_img[0,0,0]} / 255.0 = {frog_img[0,0,0] / 255.0:.6f}")

# --- Grayscale conversion: luminance formula ---
# Standard ITU-R BT.601 luma formula: Y = 0.299*R + 0.587*G + 0.114*B
# Green gets the highest weight because the human eye is most sensitive to green light
print("\nConverting RGB to Grayscale using luminance formula:")
print("  Grayscale = 0.299 * R + 0.587 * G + 0.114 * B")

frog_gray = (0.299 * frog_img[:,:,0].astype(np.float32) +
             0.587 * frog_img[:,:,1].astype(np.float32) +
             0.114 * frog_img[:,:,2].astype(np.float32))
frog_gray = frog_gray.astype(np.uint8)   # convert back to 8-bit integer

print(f"  Grayscale image shape: {frog_gray.shape}")   # (32, 32) — no channel dim

# --- Simple HSV-like visualisation: hue estimation ---
# True HSV conversion requires careful handling; here we show the concept
# by computing a simplified hue map
r_norm = frog_img[:,:,0].astype(float) / 255.0
g_norm = frog_img[:,:,1].astype(float) / 255.0
b_norm = frog_img[:,:,2].astype(float) / 255.0

# Value = maximum of R, G, B channels
value_channel = np.maximum(np.maximum(r_norm, g_norm), b_norm)

print(f"\n  Value (brightness) channel range: [{value_channel.min():.3f}, {value_channel.max():.3f}]")

# --- Visualise normalised vs grayscale ---
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle("Image Transformations — CIFAR-10 Frog", fontsize=13)

axes[0].imshow(frog_img)
axes[0].set_title("Original\nRGB uint8 [0–255]")

axes[1].imshow(frog_norm)   # matplotlib accepts float images in [0.0, 1.0]
axes[1].set_title("Normalised\nRGB float32 [0.0–1.0]")

axes[2].imshow(frog_gray, cmap='gray')
axes[2].set_title("Grayscale\n(luminance formula)")

axes[3].imshow(value_channel, cmap='plasma')
axes[3].set_title("Value Channel\n(brightness/HSV-V)")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_02_normalisation.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_02_normalisation.png")

# =====================================================================
# === SECTION 5: PIXEL HISTOGRAMS — ANALYSING IMAGE STATISTICS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 5: Pixel Histograms")
print("=" * 60)

# Compare pixel distributions across different CIFAR-10 classes
# Select one example from each of 4 visually distinct classes
class_samples = {}
classes_to_show = {0: 'airplane', 5: 'dog', 6: 'frog', 8: 'ship'}

for class_id, class_name in classes_to_show.items():
    idx = np.where(y_train_cifar == class_id)[0][0]
    class_samples[class_name] = x_train_cifar[idx]

print("\nComputing pixel histograms for: airplane, dog, frog, ship")

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle("Pixel Value Distributions Across CIFAR-10 Classes", fontsize=14)

colours = {'airplane': 'steelblue', 'dog': 'saddlebrown',
           'frog': 'green', 'ship': 'navy'}

for col_idx, (class_name, img) in enumerate(class_samples.items()):
    # Row 0: display the image
    axes[0, col_idx].imshow(img)
    axes[0, col_idx].set_title(f"Class: {class_name.capitalize()}")
    axes[0, col_idx].axis('off')

    # Row 1: per-channel histogram
    colour = colours[class_name]
    channel_colours = ['red', 'green', 'blue']

    for ch, ch_colour in enumerate(channel_colours):
        # Compute histogram for this channel: 256 bins for values 0–255
        hist, bin_edges = np.histogram(img[:, :, ch], bins=64, range=(0, 256))
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2   # centre of each bin

        # Plot as a filled line for each channel
        alpha = 0.6 if ch < 2 else 0.8   # slightly more opaque for blue
        axes[1, col_idx].plot(bin_centres, hist, color=ch_colour, alpha=alpha,
                              linewidth=1.5, label=f"Ch {ch} ({'RGB'[ch]})")
        axes[1, col_idx].fill_between(bin_centres, hist, alpha=0.15, color=ch_colour)

    axes[1, col_idx].set_xlabel("Pixel Value (0–255)")
    axes[1, col_idx].set_ylabel("Pixel Count")
    axes[1, col_idx].set_title(f"{class_name.capitalize()} — Channel Histograms")
    axes[1, col_idx].legend(fontsize=8)
    axes[1, col_idx].set_xlim(0, 255)

plt.tight_layout()
plt.savefig('lesson_02_histograms.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_02_histograms.png")

# Print statistical summary
print("\nChannel statistics for each class:")
print(f"{'Class':<12} {'R_mean':>8} {'G_mean':>8} {'B_mean':>8} {'R_std':>8} {'G_std':>8} {'B_std':>8}")
print("-" * 64)
for class_name, img in class_samples.items():
    r_mean, g_mean, b_mean = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
    r_std,  g_std,  b_std  = img[:,:,0].std(),  img[:,:,1].std(),  img[:,:,2].std()
    print(f"{class_name:<12} {r_mean:8.1f} {g_mean:8.1f} {b_mean:8.1f} "
          f"{r_std:8.1f} {g_std:8.1f} {b_std:8.1f}")

# Final summary
print("\n" + "=" * 60)
print("LESSON 02 DEMO COMPLETE")
print("=" * 60)
print("\nKey takeaways from this demo:")
print("  1. Grayscale images: shape (H, W),         dtype uint8, values 0–255")
print("  2. Colour images:    shape (H, W, 3),      dtype uint8, values 0–255")
print("  3. Batches:          shape (N, H, W, C),   float32, values 0.0–1.0")
print("  4. Normalise by dividing by 255.0 before feeding to neural networks")
print("  5. Histograms reveal colour content — frogs have high green channel values")
print("  6. Grayscale conversion uses luminance weights: R=0.299, G=0.587, B=0.114")
print("\nNext: Lesson 03 — Introduction to Neural Networks")
