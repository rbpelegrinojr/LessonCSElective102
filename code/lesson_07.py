"""
Lesson 7: Pooling Layers
CNN Image Classification Course

Sections:
  1. Manual max pooling and average pooling (NumPy)
  2. Keras MaxPooling2D and AveragePooling2D on feature maps
  3. Visualize before/after pooling (dimension reduction)
  4. Global average pooling vs Flatten for classification head
  5. Build CNN with and without pooling, compare output sizes
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, Input

print("=" * 60)
print("LESSON 7: Pooling Layers")
print("TensorFlow version:", tf.__version__)
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Manual Max Pooling and Average Pooling (NumPy)
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Manual Pooling with NumPy ---")

def max_pool_2d(feature_map, pool_size=2, stride=2):
    """Manual 2D max pooling."""
    H, W = feature_map.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = feature_map[i*stride:i*stride+pool_size,
                                 j*stride:j*stride+pool_size]
            output[i, j] = np.max(window)
    return output

def avg_pool_2d(feature_map, pool_size=2, stride=2):
    """Manual 2D average pooling."""
    H, W = feature_map.shape
    out_h = (H - pool_size) // stride + 1
    out_w = (W - pool_size) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            window = feature_map[i*stride:i*stride+pool_size,
                                 j*stride:j*stride+pool_size]
            output[i, j] = np.mean(window)
    return output

np.random.seed(42)
test_map = np.array([
    [1,  3,  2,  4,  1,  0],
    [5,  6,  7,  8,  2,  3],
    [2,  1,  9,  3,  4,  5],
    [0,  4,  2,  1,  6,  7],
    [3,  2,  1,  5,  8,  9],
    [4,  0,  3,  2,  7,  6],
], dtype=float)

print(f"  Input feature map shape: {test_map.shape}")
print("  Input feature map:")
for row in test_map:
    print("    " + "  ".join(f"{v:3.0f}" for v in row))

max_result = max_pool_2d(test_map, pool_size=2, stride=2)
avg_result = avg_pool_2d(test_map, pool_size=2, stride=2)

print(f"\n  Max Pooling result (2×2, stride 2) — shape: {max_result.shape}:")
for row in max_result:
    print("    " + "  ".join(f"{v:5.1f}" for v in row))

print(f"\n  Average Pooling result (2×2, stride 2) — shape: {avg_result.shape}:")
for row in avg_result:
    print("    " + "  ".join(f"{v:5.2f}" for v in row))

print("\n  Dimension reduction:")
print(f"    Input:  {test_map.shape[0]}×{test_map.shape[1]} = {test_map.size} values")
print(f"    After pooling: {max_result.shape[0]}×{max_result.shape[1]} = {max_result.size} values")
print(f"    Reduction factor: {test_map.size / max_result.size:.1f}×")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Keras MaxPooling2D and AveragePooling2D
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Keras Pooling Layers ---")

np.random.seed(7)
feature_map_np = np.random.rand(1, 8, 8, 1).astype('float32')
print(f"  Input feature map shape: {feature_map_np.shape}  (batch, H, W, channels)")

max_pool_layer = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
avg_pool_layer = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

max_output = max_pool_layer(feature_map_np)
avg_output = avg_pool_layer(feature_map_np)

print(f"  After MaxPooling2D(2×2, s=2):  {max_output.shape}")
print(f"  After AvgPooling2D(2×2, s=2):  {avg_output.shape}")

print("\n  Original (first 4×4 of 8×8 map):")
fm = feature_map_np[0, :4, :4, 0].numpy()
for row in fm:
    print("    " + "  ".join(f"{v:.3f}" for v in row))

print("\n  Max Pool output (2×2):")
mp = max_output[0, :2, :2, 0].numpy()
for row in mp:
    print("    " + "  ".join(f"{v:.3f}" for v in row))

print("\n  Avg Pool output (2×2):")
ap = avg_output[0, :2, :2, 0].numpy()
for row in ap:
    print("    " + "  ".join(f"{v:.3f}" for v in row))

print("\n  Testing with different pool sizes:")
for ps in [2, 3, 4]:
    layer = layers.MaxPooling2D(pool_size=(ps, ps), strides=(ps, ps))
    out = layer(feature_map_np)
    print(f"    MaxPool({ps}×{ps}, s={ps}): {feature_map_np.shape} → {out.shape}")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Visualize Before/After Pooling
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Visualizing Before/After Pooling ---")

(X_train, _), _ = tf.keras.datasets.mnist.load_data()
sample_img = X_train[0:1].astype('float32') / 255.0
sample_img = sample_img[..., np.newaxis]  # (1, 28, 28, 1)

# Create a simple edge-detection conv filter
edge_filter = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype='float32')
edge_filter = edge_filter.reshape(3, 3, 1, 1)

conv_layer = layers.Conv2D(1, (3, 3), padding='same', use_bias=False,
                            weights=[edge_filter])
feature_map_before = conv_layer(sample_img)  # (1, 28, 28, 1)

max_pool = layers.MaxPooling2D((2, 2))
avg_pool = layers.AveragePooling2D((2, 2))

after_max = max_pool(feature_map_before)   # (1, 14, 14, 1)
after_avg = avg_pool(feature_map_before)   # (1, 14, 14, 1)

print(f"  Original image shape:      {sample_img.shape[1:3]}")
print(f"  Feature map (edge) shape:  {feature_map_before.shape[1:3]}")
print(f"  After MaxPooling2D:        {after_max.shape[1:3]}")
print(f"  After AvgPooling2D:        {after_avg.shape[1:3]}")

fig, axes = plt.subplots(1, 4, figsize=(14, 3))
fig.suptitle("Pooling: Before and After", fontsize=12, fontweight='bold')

axes[0].imshow(sample_img[0, :, :, 0], cmap='gray')
axes[0].set_title(f"Original\n{sample_img.shape[1]}×{sample_img.shape[2]}")
axes[0].axis('off')

axes[1].imshow(feature_map_before[0, :, :, 0], cmap='hot')
axes[1].set_title(f"Edge Feature Map\n{feature_map_before.shape[1]}×{feature_map_before.shape[2]}")
axes[1].axis('off')

axes[2].imshow(after_max[0, :, :, 0], cmap='hot')
axes[2].set_title(f"After MaxPool (2×2)\n{after_max.shape[1]}×{after_max.shape[2]}")
axes[2].axis('off')

axes[3].imshow(after_avg[0, :, :, 0], cmap='hot')
axes[3].set_title(f"After AvgPool (2×2)\n{after_avg.shape[1]}×{after_avg.shape[2]}")
axes[3].axis('off')

plt.tight_layout()
plt.savefig("section7_pooling_visualization.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section7_pooling_visualization.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Global Average Pooling vs Flatten
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Global Average Pooling vs Flatten ---")

def build_model_gap():
    inp = Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)  # → (batch, 64)
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp, out, name="CNN_with_GAP")

def build_model_flatten():
    inp = Input(shape=(28, 28, 1))
    x = layers.Conv2D(32, 3, activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)  # → large vector
    out = layers.Dense(10, activation='softmax')(x)
    return models.Model(inp, out, name="CNN_with_Flatten")

model_gap     = build_model_gap()
model_flatten = build_model_flatten()

params_gap     = model_gap.count_params()
params_flatten = model_flatten.count_params()

print(f"  Model with GlobalAveragePooling2D: {params_gap:,} parameters")
print(f"  Model with Flatten:                {params_flatten:,} parameters")
print(f"  Parameters saved by GAP:           {params_flatten - params_gap:,}")
print(f"  Reduction ratio:                   {params_flatten / params_gap:.1f}×")

print("\n  GAP model summary:")
model_gap.summary(line_length=70)

# Verify outputs
dummy = np.random.rand(4, 28, 28, 1).astype('float32')
gap_out     = model_gap.predict(dummy, verbose=0)
flatten_out = model_flatten.predict(dummy, verbose=0)
print(f"\n  GAP model output shape:     {gap_out.shape}")
print(f"  Flatten model output shape: {flatten_out.shape}")
print("  (Both produce the same 10-class probability vector)")

# ──────────────────────────────────────────────────────────────
# SECTION 5: CNN With and Without Pooling — Compare Output Sizes
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: CNN With vs Without Pooling ---")

def build_cnn_with_pooling():
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),   # 28→14
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),   # 14→7
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),   # 7→3
        layers.Flatten(),
        layers.Dense(10, activation='softmax'),
    ], name="CNN_with_pooling")

def build_cnn_no_pooling():
    return models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(10, activation='softmax'),
    ], name="CNN_no_pooling")

model_pool   = build_cnn_with_pooling()
model_nopool = build_cnn_no_pooling()

print("  With Pooling — layer output shapes:")
for layer in model_pool.layers:
    print(f"    {layer.name:35s}: {str(layer.output_shape):30s}")

print("\n  Without Pooling — layer output shapes:")
for layer in model_nopool.layers:
    print(f"    {layer.name:35s}: {str(layer.output_shape):30s}")

print(f"\n  Model WITH pooling    — total params: {model_pool.count_params():,}")
print(f"  Model WITHOUT pooling — total params: {model_nopool.count_params():,}")

# Show flatten output sizes
flatten_pool   = 1
for d in model_pool.layers[-2].input_shape[1:]:
    flatten_pool *= d

flatten_nopool = 1
for d in model_nopool.layers[-2].input_shape[1:]:
    flatten_nopool *= d

print(f"\n  Flatten size WITH pooling:    {flatten_pool}")
print(f"  Flatten size WITHOUT pooling: {flatten_nopool}")
print(f"  Ratio: {flatten_nopool / flatten_pool:.1f}× more values without pooling")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Output Shape Progression: With vs Without Pooling", fontsize=12)

layers_pool   = [(l.name, np.prod(l.output_shape[1:])) for l in model_pool.layers]
layers_nopool = [(l.name, np.prod(l.output_shape[1:])) for l in model_nopool.layers]

names_p, sizes_p = zip(*layers_pool)
names_np, sizes_np = zip(*layers_nopool)

axes[0].bar(range(len(sizes_p)), sizes_p, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(len(names_p)))
axes[0].set_xticklabels([n[:15] for n in names_p], rotation=45, ha='right', fontsize=8)
axes[0].set_ylabel("Total Elements in Layer Output")
axes[0].set_title("With MaxPooling (log scale)")
axes[0].set_yscale('log')
axes[0].grid(True, axis='y', alpha=0.3)

axes[1].bar(range(len(sizes_np)), sizes_np, color='coral', edgecolor='black')
axes[1].set_xticks(range(len(names_np)))
axes[1].set_xticklabels([n[:15] for n in names_np], rotation=45, ha='right', fontsize=8)
axes[1].set_ylabel("Total Elements in Layer Output")
axes[1].set_title("Without Pooling (log scale)")
axes[1].set_yscale('log')
axes[1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("section7_pool_vs_no_pool.png", dpi=100, bbox_inches='tight')
plt.close()
print("  Saved: section7_pool_vs_no_pool.png")

print("\n" + "=" * 60)
print("Lesson 7 Complete!")
print("Generated images:")
print("  section7_pooling_visualization.png")
print("  section7_pool_vs_no_pool.png")
print("=" * 60)
