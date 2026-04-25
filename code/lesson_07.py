"""
Lesson 07: Pooling Layers
==========================
This module demonstrates pooling operations in CNNs:
manual max pooling and average pooling, Keras MaxPooling2D and
AveragePooling2D, Global Average Pooling, and visualization of
feature maps before and after pooling to build intuition about
dimensionality reduction and translation invariance.
"""

# === IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

print("TensorFlow version:", tf.__version__)
print("=" * 60)


# =============================================================================
# === SECTION 1: MANUAL POOLING DEMONSTRATIONS ===
# =============================================================================

print("\n[SECTION 1] Manual Max Pooling and Average Pooling")
print("-" * 60)

def manual_max_pool(feature_map, pool_size=2, stride=2):
    """
    Apply 2D max pooling manually using nested loops.
    This makes the operation transparent for learning.
    """
    h, w = feature_map.shape
    # Calculate output dimensions using the standard formula
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            # Extract the current pooling window
            window = feature_map[i*stride : i*stride + pool_size,
                                  j*stride : j*stride + pool_size]
            output[i, j] = np.max(window)  # max pooling: keep largest value

    return output

def manual_avg_pool(feature_map, pool_size=2, stride=2):
    """
    Apply 2D average pooling manually.
    Average pooling preserves overall intensity, not just peaks.
    """
    h, w = feature_map.shape
    out_h = (h - pool_size) // stride + 1
    out_w = (w - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            window = feature_map[i*stride : i*stride + pool_size,
                                  j*stride : j*stride + pool_size]
            output[i, j] = np.mean(window)  # avg pooling: mean of window

    return output

# Create a sample 4x4 feature map (simulating conv layer output)
sample_map = np.array([
    [1, 3, 2, 4],
    [5, 6, 1, 2],
    [3, 2, 5, 1],
    [1, 4, 2, 3]
], dtype=float)

max_pooled = manual_max_pool(sample_map, pool_size=2, stride=2)
avg_pooled = manual_avg_pool(sample_map, pool_size=2, stride=2)

print("  Input Feature Map (4×4):")
print(sample_map)
print(f"\n  After 2×2 Max Pooling → shape {max_pooled.shape}:")
print(max_pooled)
print(f"\n  After 2×2 Avg Pooling → shape {avg_pooled.shape}:")
print(avg_pooled)
print(f"\n  Spatial reduction: {sample_map.size} → {max_pooled.size} values "
      f"({(1 - max_pooled.size/sample_map.size)*100:.0f}% reduction)")

# Demonstrate translation invariance
print("\n  --- Translation Invariance Demo ---")
map_v1 = np.zeros((4, 4))
map_v2 = np.zeros((4, 4))
map_v1[0, 0] = 10  # feature at top-left of window
map_v2[1, 1] = 10  # feature shifted within same window

pooled_v1 = manual_max_pool(map_v1)
pooled_v2 = manual_max_pool(map_v2)

print("  Feature at position (0,0):")
print(map_v1)
print(f"  After max pool: {pooled_v1}")
print("  Feature shifted to position (1,1) (same pool window):")
print(map_v2)
print(f"  After max pool: {pooled_v2}")
print("  → Both produce the same result! This is translation invariance.")


# =============================================================================
# === SECTION 2: KERAS POOLING LAYERS ===
# =============================================================================

print("\n[SECTION 2] Keras Pooling Layers on Real Data (MNIST)")
print("-" * 60)

# Load MNIST to get real image data
(x_train, _), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0

# Add channel dimension: (N, 28, 28) → (N, 28, 28, 1)
x_train_4d = x_train[..., np.newaxis]

# Take a single sample image for visualization
sample_img = x_train_4d[0:1]  # shape: (1, 28, 28, 1)
print(f"  Input image shape: {sample_img.shape}")

# Define pooling layers using Keras
max_pool_layer = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
avg_pool_layer = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
gap_layer      = keras.layers.GlobalAveragePooling2D()

# Apply each pooling operation
output_max = max_pool_layer(sample_img).numpy()  # shape: (1, 14, 14, 1)
output_avg = avg_pool_layer(sample_img).numpy()  # shape: (1, 14, 14, 1)
output_gap = gap_layer(sample_img).numpy()        # shape: (1, 1)

print(f"  After MaxPooling2D(2,2) → shape: {output_max.shape}")
print(f"  After AvgPooling2D(2,2) → shape: {output_avg.shape}")
print(f"  After GlobalAveragePooling2D → shape: {output_gap.shape}")
print(f"  GAP output value: {output_gap[0][0]:.4f}")
print(f"  (This is the mean pixel value of the entire image)")


# =============================================================================
# === SECTION 3: FEATURE MAP SIZE COMPARISON ===
# =============================================================================

print("\n[SECTION 3] Feature Map Sizes at Different Pooling Depths")
print("-" * 60)

# Build a mini-CNN to trace spatial dimensions at each pooling step
# We use keras.layers directly (not a full model) to show individual outputs

# Create a model that outputs intermediate feature maps at each pooling stage
inputs = keras.Input(shape=(28, 28, 1))
x1 = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
p1 = keras.layers.MaxPooling2D((2, 2))(x1)    # 28×28 → 14×14

x2 = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
p2 = keras.layers.MaxPooling2D((2, 2))(x2)    # 14×14 → 7×7

x3 = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
p3 = keras.layers.MaxPooling2D((2, 2))(x3)    # 7×7 → 3×3 (floor division)

gap_out = keras.layers.GlobalAveragePooling2D()(p3)  # 3×3×64 → 64

# Create an intermediate model to extract feature maps at each stage
feature_extractor = keras.Model(
    inputs=inputs,
    outputs=[x1, p1, x2, p2, x3, p3, gap_out]
)

# Forward pass to get all intermediate feature maps
feat_outputs = feature_extractor(sample_img, training=False)

stage_names = [
    'After Conv1 (no pool)', 'After MaxPool1',
    'After Conv2 (no pool)', 'After MaxPool2',
    'After Conv3 (no pool)', 'After MaxPool3',
    'After GlobalAvgPool'
]

print(f"  {'Stage':<30} {'Shape':<25} {'Spatial Size'}")
print(f"  {'-'*30} {'-'*25} {'-'*15}")
for name, output in zip(stage_names, feat_outputs):
    shape = output.shape
    if len(shape) == 4:  # 4D tensor: (batch, h, w, channels)
        spatial = f"{shape[1]}×{shape[2]}"
        print(f"  {name:<30} {str(tuple(shape)):<25} {spatial}")
    else:  # 2D tensor after GAP
        print(f"  {name:<30} {str(tuple(shape)):<25} {'1×1 (global)'}")


# =============================================================================
# === SECTION 4: VISUALIZE POOLING EFFECTS ON MNIST IMAGE ===
# =============================================================================

print("\n[SECTION 4] Visualizing pooling effects on a real MNIST digit")
print("-" * 60)

# Get the first feature maps from our model (single channel for visualization)
feats = feature_extractor(sample_img, training=False)

# Extract spatial feature maps at each stage (use first channel only)
orig       = sample_img[0, :, :, 0]        # original 28×28 image
after_conv1 = feats[0][0, :, :, 0].numpy() # after first conv: 28×28
after_pool1 = feats[1][0, :, :, 0].numpy() # after first pool: 14×14
after_pool2 = feats[3][0, :, :, 0].numpy() # after second pool: 7×7
after_pool3 = feats[5][0, :, :, 0].numpy() # after third pool: ~3×3

fig, axes = plt.subplots(1, 5, figsize=(16, 3))
fig.suptitle('Lesson 07: Feature Map Sizes Through Pooling Layers',
             fontsize=13, fontweight='bold')

maps_to_show = [
    (orig,        'Original\n28×28'),
    (after_conv1, 'After Conv1\n28×28'),
    (after_pool1, 'After Pool1\n14×14'),
    (after_pool2, 'After Pool2\n7×7'),
    (after_pool3, 'After Pool3\n~3×3'),
]

for ax, (fmap, title) in zip(axes, maps_to_show):
    ax.imshow(fmap, cmap='viridis', interpolation='nearest')
    ax.set_title(title, fontsize=11)
    ax.axis('off')

plt.tight_layout()
plt.savefig('code/lesson_07_pooling_sizes.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_07_pooling_sizes.png")
plt.close()

# --- Visualize max vs average pooling side by side ---

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('Lesson 07: Max Pooling vs Average Pooling on MNIST Digit',
             fontsize=13, fontweight='bold')

# Show several different digits
sample_indices = [0, 1, 2, 3]

for col, idx in enumerate(sample_indices):
    img = x_train_4d[idx:idx+1]  # single image, shape (1, 28, 28, 1)
    img_max = max_pool_layer(img).numpy()[0, :, :, 0]  # 14×14
    img_avg = avg_pool_layer(img).numpy()[0, :, :, 0]  # 14×14

    # Top row: max pooled
    axes[0][col].imshow(img_max, cmap='gray', interpolation='nearest')
    axes[0][col].set_title(f'Max Pool\n(digit={y_train[idx]})', fontsize=10)
    axes[0][col].axis('off')

    # Bottom row: average pooled
    axes[1][col].imshow(img_avg, cmap='gray', interpolation='nearest')
    axes[1][col].set_title(f'Avg Pool\n(digit={y_train[idx]})', fontsize=10)
    axes[1][col].axis('off')

# Add row labels
axes[0][0].set_ylabel('Max Pooling', fontsize=11, rotation=90, labelpad=10)
axes[1][0].set_ylabel('Avg Pooling', fontsize=11, rotation=90, labelpad=10)
for ax in axes.flat:
    ax.set_ylabel('')

plt.tight_layout()
plt.savefig('code/lesson_07_max_vs_avg.png', dpi=120, bbox_inches='tight')
print("  Saved: code/lesson_07_max_vs_avg.png")
plt.close()

# --- Build a simple model comparison: max pool vs global average pool ---
print("\n  Building a CNN with MaxPool vs GlobalAveragePooling for MNIST...")

def build_pool_model(use_gap=False):
    """Build a small CNN using either MaxPool+Flatten or GlobalAveragePooling."""
    inputs = keras.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
    x = keras.layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)

    if use_gap:
        x = keras.layers.GlobalAveragePooling2D()(x)   # 28×28×32 → 32
        output_before_dense = 32
    else:
        x = keras.layers.MaxPooling2D(2, 2)(x)          # 28×28×32 → 14×14×32
        x = keras.layers.Flatten()(x)                  # 14×14×32 = 6272
        output_before_dense = 6272

    x = keras.layers.Dense(64, activation='relu')(x)
    outputs = keras.layers.Dense(10, activation='softmax')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

model_maxpool = build_pool_model(use_gap=False)
model_gap     = build_pool_model(use_gap=True)

print(f"\n  MaxPool + Flatten model parameters: "
      f"{model_maxpool.count_params():,}")
print(f"  GlobalAveragePooling model parameters: "
      f"{model_gap.count_params():,}")
print(f"  GAP reduces parameters by: "
      f"{(1 - model_gap.count_params()/model_maxpool.count_params())*100:.1f}%")

print("\n" + "=" * 60)
print("Lesson 07 complete! Generated files:")
print("  - code/lesson_07_pooling_sizes.png")
print("  - code/lesson_07_max_vs_avg.png")
print("Key takeaways:")
print("  1. Max pooling keeps the strongest feature detection in each region")
print("  2. Average pooling keeps the overall intensity in each region")
print("  3. Global Average Pooling dramatically reduces parameter count")
print("  4. Translation invariance: small feature shifts → same pool output")
print("  5. Each pooling layer halves spatial dimensions (with 2×2, stride 2)")
