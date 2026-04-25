"""
Lesson 9: Dataset Preparation & Loading
CNN Image Classification Course

Sections:
  1. Load CIFAR-10, explore statistics, visualize samples
  2. Create train/val/test splits from numpy arrays
  3. Build tf.data.Dataset pipeline with batching and prefetching
  4. Demonstrate data normalization and dtype conversion
  5. Class distribution visualization and stratified sampling
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import time

print("=" * 60)
print("LESSON 9: Dataset Preparation & Loading")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

# ──────────────────────────────────────────────────────────────
# SECTION 1: Load CIFAR-10, Explore Statistics, Visualize
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Loading and Exploring CIFAR-10 ---")

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = \
    tf.keras.datasets.cifar10.load_data()

y_train_raw = y_train_raw.flatten()
y_test_raw  = y_test_raw.flatten()

print(f"  Raw training data shape:  {X_train_raw.shape}")
print(f"  Raw test data shape:      {X_test_raw.shape}")
print(f"  Training labels shape:    {y_train_raw.shape}")
print(f"  Data type:                {X_train_raw.dtype}")
print(f"  Pixel value range:        [{X_train_raw.min()}, {X_train_raw.max()}]")
print(f"  Memory usage (train):     {X_train_raw.nbytes / 1e6:.1f} MB")

print("\n  Per-channel statistics (raw, uint8):")
for ch, name in enumerate(['Red', 'Green', 'Blue']):
    ch_data = X_train_raw[:, :, :, ch].astype(float)
    print(f"    {name}: mean={ch_data.mean():.2f}, "
          f"std={ch_data.std():.2f}, "
          f"min={ch_data.min():.0f}, max={ch_data.max():.0f}")

print("\n  Class distribution (training set):")
classes, counts = np.unique(y_train_raw, return_counts=True)
for cls, cnt in zip(classes, counts):
    bar = "█" * (cnt // 200)
    print(f"    [{cls}] {CIFAR10_CLASSES[cls]:12s}: {cnt:5d}  {bar}")

# Visualize sample images
fig, axes = plt.subplots(2, 10, figsize=(18, 4))
fig.suptitle("CIFAR-10 Sample Images (2 per class)", fontsize=12, fontweight='bold')
for cls_idx in range(10):
    class_indices = np.where(y_train_raw == cls_idx)[0]
    for row, sample_idx in enumerate(class_indices[:2]):
        axes[row, cls_idx].imshow(X_train_raw[sample_idx])
        axes[row, cls_idx].set_title(CIFAR10_CLASSES[cls_idx], fontsize=7)
        axes[row, cls_idx].axis('off')
plt.tight_layout()
plt.savefig("section9_cifar10_samples.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n  Saved: section9_cifar10_samples.png")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Create Train/Val/Test Splits
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Train / Validation / Test Splits ---")

def train_val_test_split(X, y, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split arrays into train/val/test with stratification."""
    np.random.seed(seed)
    n = len(X)
    indices = np.arange(n)

    # Stratified shuffle: sort by class then interleave
    sorted_idx = np.argsort(y)
    shuffled_idx = []
    classes_unique = np.unique(y)
    per_class = {c: sorted_idx[y[sorted_idx] == c] for c in classes_unique}
    for c in classes_unique:
        np.random.shuffle(per_class[c])

    # Build interleaved shuffled index
    all_class_idx = list(per_class.values())
    max_len = max(len(v) for v in all_class_idx)
    for i in range(max_len):
        for cls_arr in all_class_idx:
            if i < len(cls_arr):
                shuffled_idx.append(cls_arr[i])
    shuffled_idx = np.array(shuffled_idx)

    n_val  = int(n * val_ratio)
    n_test = int(n * test_ratio)

    test_idx  = shuffled_idx[:n_test]
    val_idx   = shuffled_idx[n_test:n_test + n_val]
    train_idx = shuffled_idx[n_test + n_val:]

    return (X[train_idx], y[train_idx],
            X[val_idx],   y[val_idx],
            X[test_idx],  y[test_idx])

(X_tr, y_tr,
 X_val, y_val,
 X_te, y_te) = train_val_test_split(
    X_train_raw, y_train_raw, val_ratio=0.1, test_ratio=0.1
)

print(f"  Total training data:  {len(X_train_raw):6,} samples")
print(f"  Train split:          {len(X_tr):6,} samples  ({len(X_tr)/len(X_train_raw):.0%})")
print(f"  Validation split:     {len(X_val):6,} samples  ({len(X_val)/len(X_train_raw):.0%})")
print(f"  Test split:           {len(X_te):6,} samples  ({len(X_te)/len(X_train_raw):.0%})")

print("\n  Class distribution check (should be ~balanced across splits):")
print(f"  {'Class':>12} | {'Train':>8} | {'Val':>8} | {'Test':>8}")
print("  " + "-" * 45)
for cls_idx in range(10):
    n_tr  = np.sum(y_tr  == cls_idx)
    n_val_ = np.sum(y_val == cls_idx)
    n_te  = np.sum(y_te  == cls_idx)
    print(f"  {CIFAR10_CLASSES[cls_idx]:>12} | {n_tr:>8} | {n_val_:>8} | {n_te:>8}")

print("\n  ✓ Stratified split preserves class balance across all splits")

# ──────────────────────────────────────────────────────────────
# SECTION 3: tf.data.Dataset Pipeline
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: tf.data.Dataset Pipeline ---")

BATCH_SIZE = 64

X_tr_f  = X_tr.astype('float32')  / 255.0
X_val_f = X_val.astype('float32') / 255.0
X_te_f  = X_te.astype('float32')  / 255.0

def make_dataset(X, y, batch_size, shuffle=True, cache=True):
    """Build a high-performance tf.data pipeline."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if cache:
        ds = ds.cache()
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X), reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(X_tr_f,  y_tr,  batch_size=BATCH_SIZE, shuffle=True)
val_ds   = make_dataset(X_val_f, y_val, batch_size=BATCH_SIZE, shuffle=False)
test_ds  = make_dataset(X_te_f,  y_te,  batch_size=BATCH_SIZE, shuffle=False)

print(f"  Batch size:            {BATCH_SIZE}")
print(f"  Train batches per epoch: {len(train_ds)}")
print(f"  Val batches per epoch:   {len(val_ds)}")
print(f"  Test batches:            {len(test_ds)}")

# Inspect one batch
for batch_x, batch_y in train_ds.take(1):
    print(f"\n  First batch shape:    X={batch_x.shape}, y={batch_y.shape}")
    print(f"  Batch dtype:          X={batch_x.dtype}, y={batch_y.dtype}")
    print(f"  Batch pixel range:    [{batch_x.numpy().min():.3f}, "
          f"{batch_x.numpy().max():.3f}]")

# Benchmark: with vs without prefetch
print("\n  Benchmarking: with prefetch vs without prefetch...")
def benchmark_pipeline(ds, steps=50, label=""):
    t0 = time.time()
    for i, (x, y) in enumerate(ds):
        _ = x.numpy()
        if i >= steps:
            break
    elapsed = time.time() - t0
    print(f"    {label:35s}: {elapsed:.3f}s for {steps} batches "
          f"({elapsed/steps*1000:.1f}ms/batch)")

ds_no_prefetch = (tf.data.Dataset.from_tensor_slices((X_tr_f, y_tr))
                  .shuffle(10000).batch(BATCH_SIZE))
ds_prefetch    = (tf.data.Dataset.from_tensor_slices((X_tr_f, y_tr))
                  .shuffle(10000).batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))

benchmark_pipeline(ds_no_prefetch, steps=50, label="Without prefetch")
benchmark_pipeline(ds_prefetch,    steps=50, label="With prefetch (AUTOTUNE)")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Data Normalization and dtype Conversion
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Normalization and dtype Conversion ---")

X_raw_sample = X_train_raw[:1000]

print(f"  Raw data: dtype={X_raw_sample.dtype}, "
      f"range=[{X_raw_sample.min()}, {X_raw_sample.max()}]")

# Method 1: Simple [0, 1] normalization
X_01 = X_raw_sample.astype('float32') / 255.0
print(f"\n  [0,1] normalized: dtype={X_01.dtype}, "
      f"range=[{X_01.min():.3f}, {X_01.max():.3f}], "
      f"mean={X_01.mean():.4f}")

# Method 2: ImageNet-style standardization
imagenet_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
imagenet_std  = np.array([0.229, 0.224, 0.225], dtype='float32')
X_std = (X_01 - imagenet_mean) / imagenet_std
print(f"  ImageNet standardized: dtype={X_std.dtype}, "
      f"mean={X_std.mean():.4f}, std={X_std.std():.4f}")

# Method 3: Per-batch z-score (dataset mean/std from training set)
train_mean = X_tr_f.mean(axis=(0, 1, 2), keepdims=True)
train_std  = X_tr_f.std(axis=(0, 1, 2),  keepdims=True)
X_zscore = (X_01 - train_mean) / (train_std + 1e-7)
print(f"  Z-score normalized:    dtype={X_zscore.dtype}, "
      f"mean≈{X_zscore.mean():.4f}, std≈{X_zscore.std():.4f}")

# Visualize normalization effects
fig, axes = plt.subplots(1, 4, figsize=(16, 3))
fig.suptitle("Normalization Methods Compared", fontsize=12, fontweight='bold')

sample = X_raw_sample[0]
axes[0].imshow(sample.astype(np.uint8))
axes[0].set_title(f"Raw uint8\nrange [0, 255]")
axes[0].axis('off')

axes[1].imshow(X_01[0].clip(0, 1))
axes[1].set_title(f"Divided by 255\nrange [0.0, 1.0]")
axes[1].axis('off')

# Normalize std image back to visible range for display
disp_std = (X_std[0] - X_std[0].min()) / (X_std[0].max() - X_std[0].min() + 1e-8)
axes[2].imshow(disp_std.clip(0, 1))
axes[2].set_title("ImageNet standardized\n(visualized: rescaled)")
axes[2].axis('off')

# Pixel value histograms
axes[3].hist(sample.flatten(), bins=50, alpha=0.5, label='Raw [0-255]', density=True)
axes[3].hist(X_01[0].flatten() * 255, bins=50, alpha=0.5, label='[0-1] ×255', density=True)
axes[3].set_title("Pixel Value Distributions")
axes[3].set_xlabel("Pixel value")
axes[3].legend(fontsize=8)
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("section9_normalization.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n  Saved: section9_normalization.png")

print("\n  dtype conversion summary:")
print(f"    uint8  → float32 conversion cost: "
      f"{X_raw_sample.nbytes / 1e6:.1f} MB → {X_01.nbytes / 1e6:.1f} MB "
      f"({X_01.nbytes / X_raw_sample.nbytes:.0f}× memory)")
print("    Always convert to float32 before any arithmetic operations!")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Class Distribution and Stratified Sampling
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Class Distribution and Imbalanced Data ---")

print("  Creating an artificially imbalanced dataset...")
np.random.seed(42)

# Keep all 5000 samples of classes 0-7, only 200 of classes 8-9
imbalanced_X, imbalanced_y = [], []
for cls_idx in range(10):
    mask = y_tr == cls_idx
    X_cls = X_tr_f[mask]
    y_cls = y_tr[mask]
    if cls_idx in [8, 9]:  # ship, truck: rare
        keep = np.random.choice(len(X_cls), min(200, len(X_cls)), replace=False)
        X_cls, y_cls = X_cls[keep], y_cls[keep]
    imbalanced_X.append(X_cls)
    imbalanced_y.append(y_cls)

imbalanced_X = np.concatenate(imbalanced_X)
imbalanced_y = np.concatenate(imbalanced_y)

print(f"  Imbalanced dataset size: {len(imbalanced_X)}")
print("\n  Class distribution (imbalanced):")
classes_imb, counts_imb = np.unique(imbalanced_y, return_counts=True)
for cls, cnt in zip(classes_imb, counts_imb):
    bar_len = int(cnt / max(counts_imb) * 30)
    bar = "█" * bar_len
    flag = " ← RARE CLASS" if cnt < 500 else ""
    print(f"    [{cls}] {CIFAR10_CLASSES[cls]:12s}: {cnt:5d}  {bar}{flag}")

print("\n  Computing class weights for imbalanced training:")
total = len(imbalanced_y)
class_weights = {}
for cls, cnt in zip(classes_imb, counts_imb):
    weight = total / (len(classes_imb) * cnt)
    class_weights[cls] = weight
    print(f"    Class {cls} ({CIFAR10_CLASSES[cls]:12s}): weight = {weight:.3f}")

print("\n  Usage in Keras:")
print("    model.fit(X_train, y_train, class_weight=class_weights)")

# Visualize class distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Class Distribution: Balanced vs Imbalanced", fontsize=12, fontweight='bold')

# Balanced
balanced_counts = np.array([np.sum(y_tr == i) for i in range(10)])
axes[0].bar(range(10), balanced_counts, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(10))
axes[0].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
axes[0].set_title("Balanced Dataset")
axes[0].set_ylabel("Number of Samples")
axes[0].axhline(balanced_counts.mean(), color='red', linestyle='--',
                label=f'Mean: {balanced_counts.mean():.0f}')
axes[0].legend()
axes[0].grid(True, axis='y', alpha=0.3)

# Imbalanced
imb_counts = np.array([np.sum(imbalanced_y == i) for i in range(10)])
colors = ['steelblue' if c >= 500 else 'salmon' for c in imb_counts]
axes[1].bar(range(10), imb_counts, color=colors, edgecolor='black')
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
axes[1].set_title("Imbalanced Dataset (ship/truck undersampled)")
axes[1].set_ylabel("Number of Samples")
axes[1].axhline(imb_counts.mean(), color='red', linestyle='--',
                label=f'Mean: {imb_counts.mean():.0f}')
axes[1].legend()
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("section9_class_distribution.png", dpi=100, bbox_inches='tight')
plt.close()
print("\n  Saved: section9_class_distribution.png")

# Demonstrate stratified vs random split
print("\n  Demonstrating stratified vs random split:")
np.random.seed(0)
n = len(imbalanced_y)
random_test_idx = np.random.choice(n, int(n * 0.2), replace=False)

print(f"  {'Class':>12} | {'Full %':>8} | {'Random Test %':>15} | {'Strat Test %':>14}")
print("  " + "-" * 58)
for cls_idx in range(10):
    full_pct   = np.mean(imbalanced_y == cls_idx)
    random_pct = np.mean(imbalanced_y[random_test_idx] == cls_idx)
    # Stratified: use every 5th sample of each class
    strat_pct_sum = 0
    strat_count   = 0
    for i, c in enumerate(imbalanced_y):
        if c == cls_idx and i % 5 == 0:
            strat_pct_sum += 1
            strat_count += 1
    strat_pct = strat_pct_sum / int(n * 0.2) if int(n * 0.2) > 0 else 0
    print(f"  {CIFAR10_CLASSES[cls_idx]:>12} | {full_pct:>7.2%} | "
          f"{random_pct:>14.2%} | {strat_pct:>13.2%}")

print("\n  Stratified split keeps class ratios consistent across train/test.")

print("\n" + "=" * 60)
print("Lesson 9 Complete!")
print("Generated images:")
print("  section9_cifar10_samples.png")
print("  section9_normalization.png")
print("  section9_class_distribution.png")
print("=" * 60)
