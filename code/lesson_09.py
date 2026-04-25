"""
Lesson 09: Dataset Preparation and Loading
===========================================
This module demonstrates professional data pipeline construction:
loading CIFAR-10, creating train/val/test splits, building tf.data
pipelines with batching/shuffling/prefetching, applying ImageDataGenerator
augmentation, visualizing class distributions, and exploring what
augmented images look like.
"""

# === IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time

print("TensorFlow version:", tf.__version__)
print("=" * 60)

# CIFAR-10 class names for human-readable labels
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


# =============================================================================
# === SECTION 1: LOAD CIFAR-10 AND CREATE TRAIN/VAL/TEST SPLITS ===
# =============================================================================

print("\n[SECTION 1] Loading CIFAR-10 and Creating Data Splits")
print("-" * 60)

# Keras provides CIFAR-10 with a built-in train/test split (50k/10k)
(x_train_full, y_train_full), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Flatten label arrays from (N, 1) to (N,)
y_train_full = y_train_full.flatten()
y_test       = y_test.flatten()

print(f"  Raw data loaded:")
print(f"    Training: {x_train_full.shape}, dtype={x_train_full.dtype}")
print(f"    Test:     {x_test.shape}")
print(f"    Labels range: {y_train_full.min()} to {y_train_full.max()}")

# Manually carve out a validation set from the training data
# We take the LAST 5000 samples to avoid any accidental overlap
VAL_SIZE = 5000

x_val   = x_train_full[-VAL_SIZE:]   # last 5000 samples
y_val   = y_train_full[-VAL_SIZE:]
x_train = x_train_full[:-VAL_SIZE]   # remaining 45000 samples
y_train = y_train_full[:-VAL_SIZE]

print(f"\n  After splitting:")
print(f"    Training set:   {x_train.shape}  → {len(x_train):,} samples (90%)")
print(f"    Validation set: {x_val.shape}  → {len(x_val):,}  samples  (10%)")
print(f"    Test set:       {x_test.shape}  → {len(x_test):,} samples (held out)")
print(f"\n  IMPORTANT: Test set is NEVER used until final evaluation!")

# Normalize pixel values from [0, 255] uint8 → [0.0, 1.0] float32
x_train = x_train.astype('float32') / 255.0
x_val   = x_val.astype('float32')   / 255.0
x_test  = x_test.astype('float32')  / 255.0

print(f"\n  Pixel values after normalization: min={x_train.min():.3f}, "
      f"max={x_train.max():.3f}, dtype={x_train.dtype}")


# =============================================================================
# === SECTION 2: CLASS DISTRIBUTION VISUALIZATION ===
# =============================================================================

print("\n[SECTION 2] Class Distribution Analysis")
print("-" * 60)

# Count samples per class in training set
unique_classes, train_counts = np.unique(y_train, return_counts=True)
_, val_counts  = np.unique(y_val,   return_counts=True)
_, test_counts = np.unique(y_test,  return_counts=True)

print(f"  {'Class':<12} {'Train':>8} {'Val':>8} {'Test':>8}")
print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}")
for cls_id, tc, vc, tec in zip(unique_classes, train_counts, val_counts, test_counts):
    class_name = CLASS_NAMES[cls_id]
    print(f"  {class_name:<12} {tc:>8,} {vc:>8,} {tec:>8,}")

# Check if classes are balanced
print(f"\n  Train set std dev of class counts: {train_counts.std():.1f}")
print(f"  (Low std dev = balanced dataset; CIFAR-10 is perfectly balanced)")

# Visualize class distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Lesson 09: CIFAR-10 Class Distribution',
             fontsize=14, fontweight='bold')

colors = plt.cm.Set3(np.linspace(0, 1, 10))  # 10 distinct colors

for ax, (title, counts) in zip(axes, [
    ('Training Set (45,000)', train_counts),
    ('Validation Set (5,000)', val_counts),
    ('Test Set (10,000)', test_counts)
]):
    bars = ax.bar(CLASS_NAMES, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Number of Samples')
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=np.mean(counts), color='red', linewidth=2,
               linestyle='--', label=f'Mean: {np.mean(counts):.0f}')
    ax.legend(fontsize=9)
    ax.set_ylim(0, max(counts) * 1.2)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('code/lesson_09_class_distribution.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_09_class_distribution.png")
plt.close()


# =============================================================================
# === SECTION 3: tf.data PIPELINE CONSTRUCTION ===
# =============================================================================

print("\n[SECTION 3] Building an Efficient tf.data Pipeline")
print("-" * 60)

BATCH_SIZE = 64
BUFFER_SIZE = len(x_train)  # shuffle buffer: entire training set for best mixing

# Helper: additional preprocessing (augmentation-style) using tf.data
def preprocess_train(image, label):
    """
    Preprocessing for training data.
    Uses tf ops so it runs efficiently in the tf.data pipeline.
    """
    # Random horizontal flip: mirrors image with 50% probability
    image = tf.image.random_flip_left_right(image)
    # Random brightness: simulates different lighting conditions
    image = tf.image.random_brightness(image, max_delta=0.1)
    # Clip to ensure values remain in [0, 1]
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

def preprocess_val(image, label):
    """
    Preprocessing for validation/test: NO augmentation, just return as-is.
    Augmentation on val/test would make evaluation unreliable.
    """
    return image, label

# Build training pipeline
train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))  # create dataset from arrays
    .shuffle(buffer_size=BUFFER_SIZE, seed=42)               # randomize order each epoch
    .map(preprocess_train, num_parallel_calls=tf.data.AUTOTUNE)  # augment in parallel
    .batch(BATCH_SIZE)                                        # group into batches
    .prefetch(tf.data.AUTOTUNE)                              # overlap I/O with GPU
)

# Build validation pipeline (no shuffle, no augmentation)
val_dataset = (
    tf.data.Dataset.from_tensor_slices((x_val, y_val))
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Build test pipeline
test_dataset = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
    .map(preprocess_val, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# Verify the pipeline works by extracting one batch
sample_batch_images, sample_batch_labels = next(iter(train_dataset))

print(f"  tf.data pipeline built successfully!")
print(f"  Batch image shape:  {sample_batch_images.shape}")
print(f"  Batch label shape:  {sample_batch_labels.shape}")
print(f"  Pixel value range:  [{sample_batch_images.numpy().min():.3f}, "
      f"{sample_batch_images.numpy().max():.3f}]")

# Count steps per epoch
train_steps = len(x_train) // BATCH_SIZE
val_steps   = len(x_val) // BATCH_SIZE
print(f"\n  Batch size:               {BATCH_SIZE}")
print(f"  Training steps per epoch: {train_steps}")
print(f"  Validation steps/epoch:   {val_steps}")
print(f"  (Steps = num_samples / batch_size)")

# Time how fast the pipeline is
print("\n  Timing pipeline throughput (iterating one epoch)...")
start = time.time()
count = 0
for images, labels in train_dataset:
    count += images.shape[0]  # count samples processed
elapsed = time.time() - start
print(f"  Processed {count:,} training samples in {elapsed:.2f}s "
      f"({count/elapsed:.0f} samples/sec)")


# =============================================================================
# === SECTION 4: IMAGEDATAGENERATOR AND AUGMENTATION VISUALIZATION ===
# =============================================================================

print("\n[SECTION 4] ImageDataGenerator — Augmentation Visualization")
print("-" * 60)

# Configure ImageDataGenerator with multiple augmentation types
augmentor = ImageDataGenerator(
    rotation_range=15,          # randomly rotate by up to ±15 degrees
    width_shift_range=0.1,      # randomly shift horizontally by up to 10%
    height_shift_range=0.1,     # randomly shift vertically by up to 10%
    horizontal_flip=True,       # randomly flip left-right
    zoom_range=0.1,             # randomly zoom in/out by up to 10%
    brightness_range=[0.8, 1.2], # randomly adjust brightness
    fill_mode='nearest'         # fill gaps from shifts with nearest pixel
)

# No augmentation for validation — only rescaling
val_gen_no_aug = ImageDataGenerator()

print("  Augmentation types configured:")
print("    - Random rotation:  ±15°")
print("    - Width/height shift: ±10%")
print("    - Horizontal flip")
print("    - Zoom: ±10%")
print("    - Brightness: 0.8–1.2×")

# Pick one sample image and generate 16 augmented versions
# Scale back to [0,255] range for ImageDataGenerator (it expects this)
sample_image = x_train[10:11]  # shape (1, 32, 32, 3)
sample_label = y_train[10:11]

# Generate 20 augmented versions of the same image
augmented_images = []
for aug_img, _ in augmentor.flow(sample_image * 255,  # IDG expects 0-255
                                  sample_label,
                                  batch_size=1):
    augmented_images.append(aug_img[0] / 255.0)  # normalize back to [0,1]
    if len(augmented_images) >= 15:               # stop after 15 augmentations
        break

fig, axes = plt.subplots(4, 4, figsize=(12, 12))
fig.suptitle(f'Lesson 09: Data Augmentation\n'
             f'Original: {CLASS_NAMES[sample_label[0]]} — 16 random augmentations',
             fontsize=13, fontweight='bold')

# Show original in top-left
axes[0][0].imshow(sample_image[0])
axes[0][0].set_title('ORIGINAL', fontsize=10, fontweight='bold',
                      color='blue')
axes[0][0].axis('off')

# Show 15 augmented versions
for i, (ax, aug_img) in enumerate(zip(axes.flat[1:16], augmented_images)):
    aug_img_clipped = np.clip(aug_img, 0, 1)  # ensure valid range
    ax.imshow(aug_img_clipped)
    ax.set_title(f'Augmented #{i+1}', fontsize=9)
    ax.axis('off')

plt.tight_layout()
plt.savefig('code/lesson_09_augmentation.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_09_augmentation.png")
plt.close()

# --- Visualize a batch from the pipeline ---
fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Lesson 09: Sample Batch from tf.data Training Pipeline',
             fontsize=13, fontweight='bold')

for i, ax in enumerate(axes.flat):
    if i < BATCH_SIZE // 2:  # show 32 of the 64 batch images
        img = sample_batch_images[i].numpy()
        label = sample_batch_labels[i].numpy()
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(CLASS_NAMES[label], fontsize=7)
        ax.axis('off')

plt.tight_layout()
plt.savefig('code/lesson_09_batch_samples.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_09_batch_samples.png")
plt.close()

print("\n  Summary of data pipeline:")
print(f"  Training pipeline:   shuffle → augment → batch({BATCH_SIZE}) → prefetch")
print(f"  Validation pipeline: batch({BATCH_SIZE}) → prefetch  (no shuffle/augment)")
print(f"  Test pipeline:       batch({BATCH_SIZE}) → prefetch  (no shuffle/augment)")

print("\n" + "=" * 60)
print("Lesson 09 complete! Generated files:")
print("  - code/lesson_09_class_distribution.png")
print("  - code/lesson_09_augmentation.png")
print("  - code/lesson_09_batch_samples.png")
print("Key takeaways:")
print("  1. 3 splits: train (learn) / val (tune) / test (final eval only)")
print("  2. Normalize to float32 in [0,1] for stable training")
print("  3. tf.data: shuffle → map → batch → prefetch (this order matters)")
print("  4. Augment ONLY the training set, never val or test")
print("  5. prefetch(AUTOTUNE) overlaps I/O with GPU compute")
