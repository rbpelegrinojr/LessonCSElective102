"""
Lesson 01: What Is Image Classification?
=========================================
This module demonstrates the fundamentals of image classification using the
MNIST handwritten digit dataset. We explore how computers store images as
numerical arrays, visualise individual digit samples, examine pixel value
distributions, and illustrate the basic concept of a classification pipeline.

Dataset: MNIST — 70,000 grayscale images of handwritten digits (0–9),
         each 28×28 pixels, split into 60,000 training and 10,000 test images.
"""

# === STANDARD LIBRARY & THIRD-PARTY IMPORTS ===
import numpy as np                          # numerical array operations
import matplotlib.pyplot as plt             # plotting and visualisation
import matplotlib.gridspec as gridspec      # flexible subplot layouts
from tensorflow import keras               # high-level neural network API
from tensorflow.keras.datasets import mnist  # built-in MNIST dataset loader

# =====================================================================
# === SECTION 1: LOADING AND EXPLORING THE MNIST DATASET ===
# =====================================================================

print("=" * 60)
print("SECTION 1: Loading and Exploring MNIST")
print("=" * 60)

# Load MNIST — returns two tuples: (train images, train labels) and (test images, test labels)
# Images are numpy uint8 arrays of shape (N, 28, 28); values range 0–255
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Print dataset dimensions so we understand what we are working with
print(f"\nTraining set shape:     {x_train.shape}")   # (60000, 28, 28)
print(f"Training labels shape:  {y_train.shape}")    # (60000,)
print(f"Test set shape:         {x_test.shape}")     # (10000, 28, 28)
print(f"Test labels shape:      {y_test.shape}")     # (10000,)
print(f"\nPixel value dtype:      {x_train.dtype}")  # uint8 (0–255)
print(f"Pixel value range:      {x_train.min()} – {x_train.max()}")
print(f"Unique class labels:    {np.unique(y_train)}")  # [0 1 2 3 4 5 6 7 8 9]
print(f"Number of classes:      {len(np.unique(y_train))}")

# =====================================================================
# === SECTION 2: IMAGES AS PIXEL ARRAYS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 2: Images as Pixel Arrays")
print("=" * 60)

# Select a single training image to inspect
sample_index = 0
sample_image = x_train[sample_index]   # shape: (28, 28) — a 2D numpy array
sample_label = y_train[sample_index]   # integer label, e.g., 5

print(f"\nSample image index:     {sample_index}")
print(f"True digit label:       {sample_label}")
print(f"Image array shape:      {sample_image.shape}")   # (28, 28)
print(f"Image array dtype:      {sample_image.dtype}")

# Print the raw pixel array — this is exactly what the computer stores
# Values range 0 (black background) to 255 (white ink of the digit)
print(f"\nRaw 28×28 pixel array for digit '{sample_label}':")
print(sample_image)

# Print a compact ASCII representation to show the digit's structure
print(f"\nASCII art of digit '{sample_label}' (threshold at 127):")
for row in sample_image:
    # Print '#' for ink pixels and '.' for background pixels
    print(''.join(['#' if p > 127 else '.' for p in row]))

# =====================================================================
# === SECTION 3: VISUALISING SAMPLE IMAGES FROM EACH CLASS ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 3: Visualising Sample Images")
print("=" * 60)

print("\nGenerating figure: one sample image per digit class (0–9)...")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("MNIST: One Sample Image per Class (0–9)", fontsize=16, y=1.02)

for digit in range(10):
    # Find the first occurrence of each digit in the training set
    idx = np.where(y_train == digit)[0][0]
    img = x_train[idx]

    ax = axes[digit // 5][digit % 5]   # arrange in 2 rows × 5 columns
    ax.imshow(img, cmap='gray', interpolation='nearest')
    ax.set_title(f"Digit: {digit}\nLabel: {y_train[idx]}", fontsize=10)
    ax.axis('off')  # hide axis ticks for cleaner display

plt.tight_layout()
plt.savefig('lesson_01_sample_images.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_01_sample_images.png")

# =====================================================================
# === SECTION 4: CLASS DISTRIBUTION VISUALISATION ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 4: Class Distribution Analysis")
print("=" * 60)

# Count how many training examples exist for each digit class
# np.bincount counts occurrences of each integer from 0 to max(y_train)
train_counts = np.bincount(y_train)   # shape: (10,) — one count per class
test_counts  = np.bincount(y_test)

print("\nTraining set class distribution:")
for digit, count in enumerate(train_counts):
    bar = '█' * (count // 200)   # simple ASCII bar chart in the terminal
    print(f"  Digit {digit}: {count:5d} samples  {bar}")

print(f"\nTotal training samples: {train_counts.sum()}")
print(f"Mean samples per class: {train_counts.mean():.1f}")
print(f"Std samples per class:  {train_counts.std():.1f}")
print("\nThe dataset is approximately balanced (each class has ~6,000 samples).")

# --- Matplotlib bar chart comparing train and test distributions ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("MNIST Class Distribution: Training vs Test Sets", fontsize=14)

digits = np.arange(10)
bar_width = 0.35   # width of each bar group

# Training set distribution
bars_train = ax1.bar(digits, train_counts, color='steelblue', edgecolor='black', linewidth=0.5)
ax1.set_title("Training Set (60,000 images)")
ax1.set_xlabel("Digit Class (0–9)")
ax1.set_ylabel("Number of Samples")
ax1.set_xticks(digits)
ax1.set_ylim(0, max(train_counts) * 1.15)

# Annotate each bar with its count
for bar, count in zip(bars_train, train_counts):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontsize=8)

# Test set distribution
bars_test = ax2.bar(digits, test_counts, color='coral', edgecolor='black', linewidth=0.5)
ax2.set_title("Test Set (10,000 images)")
ax2.set_xlabel("Digit Class (0–9)")
ax2.set_ylabel("Number of Samples")
ax2.set_xticks(digits)
ax2.set_ylim(0, max(test_counts) * 1.15)

for bar, count in zip(bars_test, test_counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
             str(count), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('lesson_01_class_distribution.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_01_class_distribution.png")

# --- Summary plot: pixel intensity histogram for a sample digit ---
print("\nGenerating figure: pixel intensity histogram for a sample digit...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle(f"Pixel Analysis for Digit '{sample_label}' (index {sample_index})", fontsize=13)

# Display the digit image
ax1.imshow(sample_image, cmap='gray', interpolation='nearest')
ax1.set_title(f"Digit Image (28×28 pixels)")
ax1.axis('off')

# Histogram of pixel values — shows distribution of intensities in this image
# Most pixels will be 0 (black background); a cluster near 255 represents the digit stroke
ax2.hist(sample_image.flatten(),    # flatten to 1D for histogram
         bins=32,                   # 32 bins across the 0–255 range
         color='steelblue',
         edgecolor='black',
         linewidth=0.5)
ax2.set_title("Pixel Value Distribution")
ax2.set_xlabel("Pixel Intensity (0=black, 255=white)")
ax2.set_ylabel("Number of Pixels")
ax2.axvline(127, color='red', linestyle='--', linewidth=1.5, label='Threshold (127)')
ax2.legend()

plt.tight_layout()
plt.savefig('lesson_01_pixel_histogram.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_01_pixel_histogram.png")

# =====================================================================
# === SECTION 5: DEMONSTRATING THE CLASSIFICATION CONCEPT ===
# =====================================================================

print("\n" + "=" * 60)
print("SECTION 5: Simple Classification Concept Demo")
print("=" * 60)

print("\nDemonstrating that different digits have different mean pixel values...")

# Compute the mean pixel value for each class — a very crude "feature"
# Even this naive feature separates the classes to some extent
print("\nMean pixel intensity per digit class (training set):")
for digit in range(10):
    digit_images = x_train[y_train == digit]          # select all images of this digit
    mean_intensity = digit_images.mean()               # average over all images AND pixels
    bar = '█' * int(mean_intensity / 2)
    print(f"  Digit {digit}: mean = {mean_intensity:5.2f}  {bar}")

print("\nNote: digits with more ink (8, 0) tend to have higher mean pixel values")
print("than digits with less ink (1, 7). This illustrates that pixel statistics")
print("carry class-relevant information, even before any feature extraction.")

# Show average image for each digit (the "template" for each class)
print("\nGenerating figure: average image per digit class...")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Average Image per Digit Class\n(Mean of all training examples)", fontsize=14)

for digit in range(10):
    digit_images = x_train[y_train == digit]   # all training images for this digit
    avg_image = digit_images.mean(axis=0)      # pixel-wise mean; shape (28, 28)

    ax = axes[digit // 5][digit % 5]
    ax.imshow(avg_image, cmap='gray', interpolation='nearest')
    ax.set_title(f"Digit {digit}\n({len(digit_images)} samples)", fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('lesson_01_average_digits.png', dpi=100, bbox_inches='tight')
plt.show()
print("Saved: lesson_01_average_digits.png")

# Final summary
print("\n" + "=" * 60)
print("LESSON 01 DEMO COMPLETE")
print("=" * 60)
print("\nKey takeaways from this demo:")
print("  1. Images are numpy arrays of integers (0–255 for 8-bit images)")
print("  2. MNIST grayscale images have shape (28, 28) — no colour channel")
print("  3. The dataset has 10 classes (digits 0–9), ~6,000 examples each")
print("  4. Even simple pixel statistics carry class-relevant information")
print("  5. Average images show the 'prototype' pattern for each class")
print("\nNext: Lesson 02 — Understanding Digital Images and Pixels")
