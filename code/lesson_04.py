"""
Lesson 4: From Dense Networks to CNNs
Compares fully connected networks with convolutional networks on image data.

Run: python code/lesson_04.py
Dependencies: tensorflow, numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

# =============================================================================
# SECTION 1: Count Parameters in a Dense Network for a 32×32×3 Image
# =============================================================================
print("=" * 60)
print("SECTION 1: Parameter Explosion in Dense Networks")
print("=" * 60)
print("\nA 32×32×3 color image (CIFAR-10 size) = 3,072 raw input values.")
print("Watch what happens as we add dense layers...\n")

INPUT_SIZE = 32 * 32 * 3  # = 3072

def count_dense_params(layer_sizes):
    """Calculate total parameters for a fully connected network."""
    total = 0
    details = []
    for i in range(1, len(layer_sizes)):
        w = layer_sizes[i - 1] * layer_sizes[i]
        b = layer_sizes[i]
        layer_total = w + b
        total += layer_total
        details.append({
            "from": layer_sizes[i - 1],
            "to": layer_sizes[i],
            "weights": w,
            "biases": b,
            "layer_total": layer_total,
        })
    return total, details

architectures = [
    [INPUT_SIZE, 512, 10],
    [INPUT_SIZE, 1024, 512, 10],
    [INPUT_SIZE, 2048, 1024, 512, 256, 10],
    [INPUT_SIZE, 4096, 2048, 1024, 10],
]

print(f"{'Architecture':<45} {'Total Params':>15} {'Memory (MB)':>12}")
print("-" * 75)
param_counts_dense = []
for arch in architectures:
    total, details = count_dense_params(arch)
    mem_mb = total * 4 / (1024 ** 2)
    arch_str = " → ".join(str(s) for s in arch)
    print(f"{arch_str:<45} {total:>15,} {mem_mb:>11.1f}")
    param_counts_dense.append(total)

print("\nDetailed breakdown for architecture [3072 → 1024 → 512 → 10]:")
total, details = count_dense_params([INPUT_SIZE, 1024, 512, 10])
for i, d in enumerate(details):
    print(f"  Layer {i+1}: {d['from']} × {d['to']} weights + {d['to']} biases = {d['layer_total']:,}")
print(f"  TOTAL: {total:,} parameters")

print(f"\nFor reference:")
print(f"  1 million parameters at float32 = {4 * 1e6 / (1024**2):.1f} MB of memory")
print(f"  AlexNet total params            ≈ 62,000,000")
print(f"  ResNet-50 total params          ≈ 25,000,000")
print(f"  Your 4-layer dense net          = {param_counts_dense[2]:,} params")

# =============================================================================
# SECTION 2: Count Parameters in an Equivalent CNN
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: Parameter Efficiency of CNNs")
print("=" * 60)
print("\nCNNs use parameter sharing — one filter scans the whole image.")
print("The number of parameters is determined by filter size, not image size.\n")

def count_conv_params(in_channels, out_channels, kernel_size):
    """Parameters in a Conv2D layer (no bias separate)."""
    weights = kernel_size * kernel_size * in_channels * out_channels
    biases = out_channels
    return weights + biases, weights, biases

print("Conv2D layer parameter counts:")
print(f"  {'Config':<45} {'Params':>10} {'vs Dense':>15}")
print("-" * 75)

dense_first_layer = INPUT_SIZE * 512
conv_configs = [
    (3, 32, 3),
    (3, 64, 3),
    (1, 32, 3),
    (3, 32, 5),
    (64, 128, 3),
]
for in_c, out_c, k in conv_configs:
    total, w, b = count_conv_params(in_c, out_c, k)
    config_str = f"{k}×{k} kernel, {in_c} in_ch → {out_c} filters"
    ratio = dense_first_layer / total
    print(f"  {config_str:<45} {total:>10,} {ratio:>13.0f}× smaller")

print("\nA full simple CNN vs a full dense network:")
cnn_total = 0
cnn_layers = [
    ("Conv(3, 32 filters, 3×3)", *count_conv_params(3, 32, 3)),
    ("Conv(32, 64 filters, 3×3)", *count_conv_params(32, 64, 3)),
    ("Conv(64, 128 filters, 3×3)", *count_conv_params(64, 128, 3)),
    ("Dense(128×4×4 → 256)", 128 * 4 * 4 * 256 + 256, 128 * 4 * 4 * 256, 256),
    ("Dense(256 → 10)", 256 * 10 + 10, 256 * 10, 10),
]

for name, total, _, _ in cnn_layers:
    cnn_total += total
    print(f"  {name:<40} {total:>10,}")
print(f"  {'CNN TOTAL':<40} {cnn_total:>10,}")

dense_total = param_counts_dense[1]
print(f"\n  Dense network total  : {dense_total:,}")
print(f"  CNN total            : {cnn_total:,}")
print(f"  CNN is {dense_total // cnn_total}× more parameter-efficient!")

# =============================================================================
# SECTION 3: Spatial Invariance Demonstration
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: Spatial Invariance — Same Feature, Different Location")
print("=" * 60)
print("\nA CNN detects the same feature wherever it appears in an image.")
print("A dense network treats different positions as completely different inputs.\n")

SIZE = 16

def create_feature_image(position, size=SIZE):
    """Create an image with a small cross feature at a given position."""
    img = np.zeros((size, size), dtype=np.float32)
    r, c = position
    if 1 <= r <= size - 2 and 1 <= c <= size - 2:
        img[r - 1:r + 2, c] = 1.0
        img[r, c - 1:c + 2] = 1.0
    return img

positions = [(2, 2), (2, 13), (7, 7), (13, 2), (13, 13)]
images = [create_feature_image(p) for p in positions]

edge_kernel = np.array([[-1, -1, -1],
                         [-1,  8, -1],
                         [-1, -1, -1]], dtype=np.float32)

def manual_conv2d(image, kernel, padding=1):
    h, w = image.shape
    kh, kw = kernel.shape
    padded = np.pad(image, padding, mode="constant")
    out_h = h - kh + 1 + 2 * padding
    out_w = w - kw + 1 + 2 * padding
    output = np.zeros((out_h, out_w))
    for i in range(out_h):
        for j in range(out_w):
            patch = padded[i:i + kh, j:j + kw]
            output[i, j] = (patch * kernel).sum()
    return output

feature_maps = [manual_conv2d(img, edge_kernel) for img in images]

dense_flat = [img.flatten() for img in images]
cosine_sims = []
for i in range(len(dense_flat)):
    for j in range(i + 1, len(dense_flat)):
        dot = np.dot(dense_flat[i], dense_flat[j])
        norm = np.linalg.norm(dense_flat[i]) * np.linalg.norm(dense_flat[j])
        sim = dot / (norm + 1e-9)
        cosine_sims.append(sim)

conv_responses = [fm.max() for fm in feature_maps]

print("Result — same cross feature at 5 different positions:")
print(f"  {'Position':<15} {'Conv max response':>20} {'Dense similarity to img[0]':>28}")
print("-" * 65)
sim_idx = 0
for i, (pos, resp) in enumerate(zip(positions, conv_responses)):
    if i == 0:
        dense_note = "(reference)"
    else:
        dense_note = f"{cosine_sims[sim_idx - 1]:.4f}"
        sim_idx += 1
    print(f"  {str(pos):<15} {resp:>20.4f} {dense_note:>28}")

print(f"\n  CNN: max response is ~{np.mean(conv_responses):.2f} everywhere → spatially invariant!")
print(f"  Dense: cosine similarity to reference = {np.mean(cosine_sims):.4f} → different inputs!")

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle("SECTION 3: Spatial Invariance — Same Feature at Different Locations\n"
             "Top: Input images  |  Bottom: Conv feature maps (similar response everywhere)",
             fontsize=11, fontweight="bold")
for i, (img, fm, pos) in enumerate(zip(images, feature_maps, positions)):
    axes[0, i].imshow(img, cmap="gray", vmin=0, vmax=1)
    axes[0, i].set_title(f"Position {pos}", fontsize=9)
    axes[0, i].axis("off")
    axes[1, i].imshow(fm, cmap="hot")
    axes[1, i].set_title(f"Max: {fm.max():.1f}", fontsize=9)
    axes[1, i].axis("off")
plt.tight_layout()
plt.savefig("section3_spatial_invariance.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section3_spatial_invariance.png]")

# =============================================================================
# SECTION 4: Dense Network for CIFAR-10 — Parameter Count and Limitations
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: Dense Network for CIFAR-10 — Build and Inspect")
print("=" * 60)
print("\nBuilding a dense-only network for CIFAR-10 (32×32×3 images, 10 classes).")
print("We will count parameters and show why this approach is limited.\n")

dense_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation="relu"),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
], name="Dense_CIFAR10")

dense_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Dense model summary:")
dense_model.summary()

total_dense = dense_model.count_params()
mem_dense_mb = total_dense * 4 / (1024 ** 2)
print(f"\nTotal parameters : {total_dense:,}")
print(f"Memory for weights: {mem_dense_mb:.1f} MB")
print("\nLimitations of this approach:")
print("  1. Huge parameter count → slow training, needs lots of data")
print("  2. No spatial awareness → pixel (0,0) treated same as (31,31)")
print("  3. Cannot detect same feature at different locations efficiently")
print("  4. Prone to overfitting on limited training data")

# =============================================================================
# SECTION 5: Simple CNN for Same Task — Compare Parameter Counts
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: Simple CNN for CIFAR-10 — Parameter Comparison")
print("=" * 60)
print("\nNow let us build a CNN for the same task and compare.\n")

cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
], name="CNN_CIFAR10")

cnn_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("CNN model summary:")
cnn_model.summary()

total_cnn = cnn_model.count_params()
mem_cnn_mb = total_cnn * 4 / (1024 ** 2)
print(f"\nTotal parameters : {total_cnn:,}")
print(f"Memory for weights: {mem_cnn_mb:.1f} MB")

print("\n" + "─" * 50)
print(f"{'Model':<20} {'Parameters':>15} {'Memory':>12}")
print("─" * 50)
print(f"{'Dense':<20} {total_dense:>15,} {mem_dense_mb:>10.1f} MB")
print(f"{'CNN':<20} {total_cnn:>15,} {mem_cnn_mb:>10.1f} MB")
print(f"{'Reduction':<20} {total_dense // total_cnn:>14}× {' ':>12}")
print("─" * 50)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("SECTION 5: Dense vs CNN — Parameter Count Comparison",
             fontsize=13, fontweight="bold")

model_names = ["Dense\n(3-hidden layers)", "CNN\n(3 conv + 1 dense)"]
param_totals = [total_dense, total_cnn]
colors = ["tomato", "steelblue"]

bars = axes[0].bar(model_names, param_totals, color=colors, edgecolor="black", width=0.4)
axes[0].set_title("Total Trainable Parameters", fontsize=11)
axes[0].set_ylabel("Number of Parameters")
for bar, count in zip(bars, param_totals):
    axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                 f"{count:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")

layer_names_dense = ["Flatten→1024", "1024→512", "512→256", "256→10"]
layer_params_dense = [3072 * 1024 + 1024, 1024 * 512 + 512, 512 * 256 + 256, 256 * 10 + 10]

layer_names_cnn = ["Conv(3→32)", "Conv(32→64)", "Conv(64→128)", "FC(512→256)", "FC(256→10)"]
fc_input_size = (32 // 8) * (32 // 8) * 128
layer_params_cnn = [
    count_conv_params(3, 32, 3)[0],
    count_conv_params(32, 64, 3)[0],
    count_conv_params(64, 128, 3)[0],
    fc_input_size * 256 + 256,
    256 * 10 + 10,
]

x = np.arange(max(len(layer_names_dense), len(layer_names_cnn)))
axes[1].bar(np.arange(len(layer_names_dense)) - 0.2, layer_params_dense,
            width=0.35, color="tomato", label="Dense", edgecolor="black", alpha=0.8)
axes[1].bar(np.arange(len(layer_params_cnn)) + 0.2, layer_params_cnn,
            width=0.35, color="steelblue", label="CNN", edgecolor="black", alpha=0.8)
axes[1].set_title("Parameters per Layer", fontsize=11)
axes[1].set_ylabel("Parameters")
axes[1].set_yscale("log")
axes[1].legend()
axes[1].set_xticks(np.arange(max(len(layer_names_dense), len(layer_params_cnn))))
axes[1].set_xticklabels([f"Layer {i+1}" for i in range(5)], rotation=30)

plt.tight_layout()
plt.savefig("section5_param_comparison.png", dpi=100, bbox_inches="tight")
print("\n[Saved: section5_param_comparison.png]")

print("\n" + "=" * 60)
print("All 5 sections complete!")
print("Generated files:")
for fname in ["section3_spatial_invariance.png", "section5_param_comparison.png"]:
    print(f"  {fname}")
print("=" * 60)

plt.show()
