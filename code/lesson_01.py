"""
Lesson 1: What Is Image Classification?
Demonstrates the basics of image classification using the MNIST dataset.

Run: python code/lesson_01.py
Dependencies: tensorflow, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# SECTION 1: Loading and Exploring a Real Dataset
# =============================================================================
print("=" * 60)
print("SECTION 1: Loading and Exploring the MNIST Dataset")
print("=" * 60)
print("\nMNIST = Modified National Institute of Standards and Technology")
print("It contains 70,000 grayscale 28x28 images of handwritten digits (0-9).")
print("This is the 'Hello, World!' of image classification.\n")

from tensorflow.keras.datasets import mnist

try:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print("MNIST loaded from Keras datasets.")
except Exception as e:
    print(f"Could not download MNIST ({e}).")
    print("Generating synthetic MNIST-like data for demonstration.\n")
    rng_data = np.random.default_rng(0)
    x_train = rng_data.integers(0, 256, (60000, 28, 28), dtype=np.uint8)
    y_train = rng_data.integers(0, 10, 60000)
    x_test  = rng_data.integers(0, 256, (10000, 28, 28), dtype=np.uint8)
    y_test  = rng_data.integers(0, 10, 10000)

print(f"Training images shape : {x_train.shape}")
print(f"  → {x_train.shape[0]} images, each {x_train.shape[1]}x{x_train.shape[2]} pixels")
print(f"Training labels shape : {y_train.shape}")
print(f"Test images shape     : {x_test.shape}")
print(f"Test labels shape     : {y_test.shape}")
print(f"\nPixel value range     : {x_train.min()} to {x_train.max()}")
print(f"Data type             : {x_train.dtype}")
print(f"Number of classes     : {len(np.unique(y_train))} ({np.unique(y_train)})")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("SECTION 1: Sample MNIST Images with Labels", fontsize=14, fontweight="bold")
for i, ax in enumerate(axes.flat):
    ax.imshow(x_train[i], cmap="gray")
    ax.set_title(f"Label: {y_train[i]}", fontsize=12)
    ax.axis("off")
plt.tight_layout()
plt.savefig("section1_mnist_samples.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section1_mnist_samples.png]")

# =============================================================================
# SECTION 2: What Does an Image Look Like to a Computer?
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: What Does an Image Look Like to a Computer?")
print("=" * 60)
print("\nTo a human, an MNIST image looks like a digit.")
print("To a computer, it is just a 28x28 grid of numbers.\n")

sample_image = x_train[0]
sample_label = y_train[0]

print(f"This image has label: {sample_label}")
print(f"Image shape: {sample_image.shape} — that is {sample_image.size} numbers\n")
print("Pixel values (0 = black, 255 = white):")
print("-" * 60)

for row in sample_image[6:22:2]:
    row_str = " ".join(f"{val:3d}" for val in row[4:24:2])
    print(row_str)

print("-" * 60)
print("(Showing every other row/column for readability)\n")

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("SECTION 2: Image as Pixel Array", fontsize=14, fontweight="bold")

axes[0].imshow(sample_image, cmap="gray")
axes[0].set_title(f"Visual Image (Label: {sample_label})", fontsize=11)
axes[0].axis("off")

im = axes[1].imshow(sample_image, cmap="viridis")
axes[1].set_title("Heatmap (pixel intensity)", fontsize=11)
plt.colorbar(im, ax=axes[1])

axes[2].imshow(sample_image[8:20, 8:20], cmap="gray")
axes[2].set_title("Zoomed center (12×12)", fontsize=11)
for y_pos in range(12):
    for x_pos in range(12):
        val = sample_image[8 + y_pos, 8 + x_pos]
        axes[2].text(x_pos, y_pos, str(val), ha="center", va="center",
                     fontsize=5, color="white" if val < 128 else "black")

plt.tight_layout()
plt.savefig("section2_pixel_view.png", dpi=100, bbox_inches="tight")
print("[Saved: section2_pixel_view.png]")

# =============================================================================
# SECTION 3: The Classification Pipeline
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: The Classification Pipeline")
print("=" * 60)
print("\nA classifier outputs one probability per class.")
print("The class with the highest probability is the prediction.")
print("This is called the argmax of the probability vector.\n")

def softmax(logits):
    """Convert raw scores to probabilities (sum to 1.0)."""
    exp_vals = np.exp(logits - np.max(logits))
    return exp_vals / exp_vals.sum()

class_names = [str(i) for i in range(10)]

rng = np.random.default_rng(seed=42)
untrained_logits = rng.normal(0, 1, size=10)
untrained_probs = softmax(untrained_logits)

print("Untrained (random) model output for one MNIST image:")
print(f"  Raw logits  : {np.round(untrained_logits, 3)}")
print(f"  Probabilities: {np.round(untrained_probs, 3)}")
print(f"  Sum of probs : {untrained_probs.sum():.4f}  ← must equal 1.0")
print(f"  Predicted class: {np.argmax(untrained_probs)} "
      f"(confidence: {untrained_probs.max():.1%})")
print(f"  Actual label   : {sample_label}")
print("\nNote: Untrained model guesses randomly. Training will fix this.\n")

trained_logits = np.array([-2.1, -3.5, -1.2, -0.8, 8.7, -2.0, -3.1, -0.5, -1.9, -2.3])
trained_probs = softmax(trained_logits)
print("Trained model output for the same image:")
print(f"  Probabilities: {np.round(trained_probs, 3)}")
print(f"  Predicted class: {np.argmax(trained_probs)} "
      f"(confidence: {trained_probs.max():.1%})")
print(f"  Actual label   : {sample_label}")
print("\n  ✓ After training, the model is highly confident and correct!")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("SECTION 3: Classification Pipeline — Probability Outputs",
             fontsize=13, fontweight="bold")

axes[0].bar(class_names, untrained_probs, color="tomato", edgecolor="black")
axes[0].set_title("Untrained Model (Random Guessing)", fontsize=11)
axes[0].set_xlabel("Digit Class")
axes[0].set_ylabel("Probability")
axes[0].set_ylim(0, 1)
axes[0].axhline(0.1, color="gray", linestyle="--", alpha=0.5, label="Chance level")
axes[0].legend()

axes[1].bar(class_names, trained_probs, color="steelblue", edgecolor="black")
axes[1].set_title(f"Trained Model (Predicted: {np.argmax(trained_probs)}, "
                  f"Actual: {sample_label})", fontsize=11)
axes[1].set_xlabel("Digit Class")
axes[1].set_ylabel("Probability")
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig("section3_pipeline.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section3_pipeline.png]")

# =============================================================================
# SECTION 4: Exploring Class Distribution
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Exploring Class Distribution")
print("=" * 60)
print("\nA well-balanced dataset has roughly equal numbers per class.")
print("Class imbalance can bias a model toward the majority class.\n")

unique_classes, counts = np.unique(y_train, return_counts=True)
print("Training set class distribution:")
print(f"  {'Digit':<8} {'Count':<8} {'Percentage'}")
print("  " + "-" * 35)
for cls, cnt in zip(unique_classes, counts):
    bar = "█" * (cnt // 300)
    pct = cnt / len(y_train) * 100
    print(f"  {cls:<8} {cnt:<8} {pct:.1f}%  {bar}")

print(f"\n  Total training examples : {len(y_train):,}")
print(f"  Most common class       : {unique_classes[np.argmax(counts)]} "
      f"({counts.max():,} examples)")
print(f"  Least common class      : {unique_classes[np.argmin(counts)]} "
      f"({counts.min():,} examples)")
print(f"  Max/Min ratio           : {counts.max() / counts.min():.2f}x")
print("\n  → MNIST is very well balanced. All classes have ~6,000 examples.")

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(unique_classes, counts, color=plt.cm.tab10(unique_classes / 10.0),
              edgecolor="black", width=0.7)
ax.set_title("SECTION 4: Class Distribution in MNIST Training Set",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Digit Class")
ax.set_ylabel("Number of Examples")
ax.set_xticks(unique_classes)
ax.axhline(counts.mean(), color="red", linestyle="--",
           label=f"Mean: {counts.mean():.0f}", linewidth=2)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
            str(count), ha="center", va="bottom", fontsize=10)
ax.legend()
plt.tight_layout()
plt.savefig("section4_class_distribution.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section4_class_distribution.png]")

# =============================================================================
# SECTION 5: Visualizing Multiple Classes
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Visualizing Multiple Classes (One of Each Digit)")
print("=" * 60)
print("\nLet's look at one example of each digit class side by side.")
print("This shows the variety within the dataset.\n")

fig, axes = plt.subplots(2, 10, figsize=(20, 5))
fig.suptitle("SECTION 5: One Example of Each Digit Class (0–9)\n"
             "Top row: First occurrence  |  Bottom row: A different occurrence",
             fontsize=12, fontweight="bold")

first_occurrence = {}
second_occurrence = {}

for idx, label in enumerate(y_train):
    if label not in first_occurrence:
        first_occurrence[label] = idx
    elif label not in second_occurrence:
        second_occurrence[label] = idx
    if len(second_occurrence) == 10:
        break

for digit in range(10):
    idx1 = first_occurrence[digit]
    idx2 = second_occurrence[digit]

    axes[0, digit].imshow(x_train[idx1], cmap="gray")
    axes[0, digit].set_title(f"Digit {digit}", fontsize=11, fontweight="bold")
    axes[0, digit].axis("off")

    axes[1, digit].imshow(x_train[idx2], cmap="gray")
    axes[1, digit].set_title(f"Digit {digit}", fontsize=11)
    axes[1, digit].axis("off")

plt.tight_layout()
plt.savefig("section5_all_classes.png", dpi=100, bbox_inches="tight")
print("Grid of 20 images (2 examples × 10 digit classes) created.")
print("[Saved: section5_all_classes.png]")

print("\n" + "=" * 60)
print("All 5 sections complete!")
print("Generated files:")
print("  section1_mnist_samples.png")
print("  section2_pixel_view.png")
print("  section3_pipeline.png")
print("  section4_class_distribution.png")
print("  section5_all_classes.png")
print("=" * 60)

plt.show()
