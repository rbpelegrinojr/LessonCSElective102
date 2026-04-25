"""
Lesson 18: Model Visualization and Interpretability with Grad-CAM
==================================================================
This module implements interpretability techniques for CNNs:
  - Feature map visualisation at different network depths
  - Grad-CAM (Gradient-weighted Class Activation Mapping) from scratch
  - Saliency map computation
  - Occlusion sensitivity maps
  - Side-by-side comparison of techniques and classes
"""

# === STANDARD LIBRARY IMPORTS ===
import os

# === NUMERICAL / PLOTTING IMPORTS ===
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# === TENSORFLOW / KERAS IMPORTS ===
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 60)
print("LESSON 18: Model Visualisation and Interpretability")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")


# ===================================================================
# SECTION 1: SETUP — LOAD / TRAIN MODEL, PREPARE TEST IMAGES
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 1: Setup — Load/Build Model, Prepare Test Images")
print("=" * 60)

CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]
NUM_CLASSES = 10
IMG_SIZE    = 96
BATCH_SIZE  = 64

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train_flat = y_train.squeeze()   # (50000,)
y_test_flat  = y_test.squeeze()    # (10000,)

def preprocess_image(img):
    """Resize single image (H×W×C uint8) and apply MobileNetV2 normalisation."""
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = mobilenet_preprocess(img)   # scale to [-1, 1]
    return img.numpy()

def preprocess_dataset(x, y):
    """Build a tf.data.Dataset."""
    def _preprocess(images, labels):
        images = tf.cast(images, tf.float32)
        images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
        images = mobilenet_preprocess(images)
        labels = tf.squeeze(labels, axis=-1)
        labels = tf.one_hot(labels, NUM_CLASSES)
        return images, labels
    return (tf.data.Dataset.from_tensor_slices((x, y))
            .map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

# --- Try to load a fine-tuned model from Lesson 17 or 16 ---
MODEL_PATHS = ['lesson_17_finetuned_model.keras', 'lesson_16_feature_extraction_model.keras']
model = None
for path in MODEL_PATHS:
    if os.path.exists(path):
        print(f"Loading model from '{path}'...")
        model = keras.models.load_model(path)
        print("Model loaded successfully.")
        break

if model is None:
    print("No saved model found. Training a quick model for demonstration...")
    TRAIN_N = 8000
    base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       include_top=False, weights='imagenet')
    base.trainable = False
    inputs   = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x        = base(inputs, training=False)
    x        = layers.GlobalAveragePooling2D()(x)
    x        = layers.Dense(256, activation='relu')(x)
    x        = layers.Dropout(0.5)(x)
    outputs  = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    model    = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    train_ds_quick = preprocess_dataset(x_train[:TRAIN_N], y_train[:TRAIN_N])
    val_ds_quick   = preprocess_dataset(x_train[TRAIN_N:TRAIN_N+1000],
                                        y_train[TRAIN_N:TRAIN_N+1000])
    model.fit(train_ds_quick, validation_data=val_ds_quick, epochs=10,
              callbacks=[keras.callbacks.EarlyStopping(
                  monitor='val_accuracy', patience=3, restore_best_weights=True)],
              verbose=1)

test_ds = preprocess_dataset(x_test, y_test)
_, test_acc = model.evaluate(test_ds, verbose=0)
print(f"Model test accuracy: {test_acc*100:.2f}%")

# Select 5 test images for visualisation (one per class variety)
VIS_INDICES = [0, 10, 50, 100, 300]   # varied CIFAR-10 test samples
vis_images_raw  = x_test[VIS_INDICES]            # original uint8 for display
vis_labels      = y_test_flat[VIS_INDICES]
vis_images_proc = np.stack([preprocess_image(img) for img in vis_images_raw])
print(f"\nSelected {len(VIS_INDICES)} test images for visualisation:")
for i, (idx, lbl) in enumerate(zip(VIS_INDICES, vis_labels)):
    preds = model.predict(vis_images_proc[i:i+1], verbose=0)
    pred_class = np.argmax(preds)
    print(f"  Image {i}: true={CLASS_NAMES[lbl]:12s} pred={CLASS_NAMES[pred_class]:12s} "
          f"conf={preds[0][pred_class]:.3f}")


# ===================================================================
# SECTION 2: FEATURE MAP VISUALISATION
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 2: Feature Map Visualisation")
print("=" * 60)

# Find convolutional layers in the model's base network
def get_conv_layers(model):
    """Collect Conv2D layer names at different depths."""
    conv_layers = []
    for layer in model.layers:
        if hasattr(layer, 'layers'):          # nested model (MobileNetV2 base)
            for sub_layer in layer.layers:
                if isinstance(sub_layer, layers.Conv2D):
                    conv_layers.append((layer.name + '/' + sub_layer.name, sub_layer))
        elif isinstance(layer, layers.Conv2D):
            conv_layers.append((layer.name, layer))
    return conv_layers

conv_layers = get_conv_layers(model)
print(f"Found {len(conv_layers)} Conv2D layers in the model.")

# Build intermediate output model for selected layers
# We sample 3 layers: first, middle, and near-last conv
selected_indices = [0,
                    len(conv_layers) // 2,
                    min(len(conv_layers) - 1, len(conv_layers) - 3)]
selected_layers  = [conv_layers[i] for i in selected_indices]

# Find these layers inside the MobileNetV2 base sub-model
mn_base_layer = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name:
        mn_base_layer = layer
        break

if mn_base_layer is not None:
    # Build a model that outputs feature maps from early and late sub-layers
    # Use a small sample from the base
    sample_sub_layers = []
    for sub_layer in mn_base_layer.layers:
        if isinstance(sub_layer, layers.Conv2D):
            sample_sub_layers.append(sub_layer)

    if len(sample_sub_layers) >= 3:
        layer_targets = [
            sample_sub_layers[0],                              # first conv layer
            sample_sub_layers[len(sample_sub_layers) // 2],   # middle conv layer
            sample_sub_layers[-1],                             # last conv layer
        ]
        depth_labels = ['Early Layer (Edges)', 'Middle Layer (Textures)', 'Late Layer (Parts)']

        # Build sub-model with multiple outputs
        feature_model = keras.Model(
            inputs=mn_base_layer.input,
            outputs=[l.output for l in layer_targets]
        )

        # Run the first vis image through the base
        img_input = vis_images_proc[0:1]                # shape (1, 96, 96, 3)
        # Get the input to the base (after the first Input layer of our model)
        base_input = model.layers[1].input               # the base layer input
        base_output_fn = keras.backend.function(
            [model.input],
            [mn_base_layer(model.input)]
        )

        # Directly call feature_model on the preprocessed image
        all_feature_maps = feature_model(img_input, training=False)

        print(f"\nFeature map shapes:")
        for label, fmaps in zip(depth_labels, all_feature_maps):
            print(f"  {label}: {fmaps.shape}")

        # Visualise first 16 feature maps from each depth
        fig, big_axes = plt.subplots(3, 1, figsize=(14, 12))
        fig.suptitle('Feature Map Visualisation Across Network Depth', fontsize=14, fontweight='bold')

        for row_idx, (label, fmaps) in enumerate(zip(depth_labels, all_feature_maps)):
            fmaps_np = fmaps.numpy()[0]    # shape (H, W, C); take batch index 0
            n_show   = min(16, fmaps_np.shape[-1])

            # Create a grid of 4×4 (or smaller) feature maps
            cols = 8
            rows_inner = (n_show + cols - 1) // cols
            grid_h = rows_inner * fmaps_np.shape[0] + (rows_inner - 1) * 2
            grid_w = cols * fmaps_np.shape[1] + (cols - 1) * 2
            canvas = np.zeros((grid_h, grid_w))

            for k in range(n_show):
                r = k // cols
                c = k  % cols
                fh, fw = fmaps_np.shape[0], fmaps_np.shape[1]
                r_start = r * (fh + 2)
                c_start = c * (fw + 2)
                # Normalise each map to [0,1] for display
                fm = fmaps_np[:, :, k]
                fm = (fm - fm.min()) / (fm.max() - fm.min() + 1e-8)
                canvas[r_start:r_start+fh, c_start:c_start+fw] = fm

            big_axes[row_idx].imshow(canvas, cmap='viridis', aspect='auto')
            big_axes[row_idx].set_title(f'{label}  (showing {n_show} of '
                                        f'{fmaps_np.shape[-1]} feature maps)',
                                        fontsize=11)
            big_axes[row_idx].axis('off')

        plt.tight_layout()
        plt.savefig('lesson_18_feature_maps.png', dpi=80, bbox_inches='tight')
        plt.close()
        print("\nSaved: lesson_18_feature_maps.png")
    else:
        print("Not enough conv sub-layers found; skipping feature map grid.")
else:
    print("MobileNetV2 base not found; skipping feature map grid.")


# ===================================================================
# SECTION 3: GRAD-CAM IMPLEMENTATION FROM SCRATCH
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 3: Grad-CAM Implementation from Scratch")
print("=" * 60)

def make_gradcam_heatmap(model, img_array, last_conv_layer_name, pred_index=None):
    """
    Compute a Grad-CAM heatmap for the given image.

    Algorithm:
      1. Build a model that outputs both the last conv layer activations
         AND the final predictions.
      2. Record the gradient of the top class score with respect to the
         last conv layer using GradientTape.
      3. Pool gradients spatially → importance weights per feature map.
      4. Compute weighted sum of feature maps, apply ReLU.
      5. Normalise to [0, 1].

    Parameters
    ----------
    model            : trained Keras model
    img_array        : preprocessed image, shape (1, H, W, 3)
    last_conv_layer_name : name of the last convolutional layer to target
    pred_index       : class index for which to compute Grad-CAM;
                       if None, uses the argmax of the prediction

    Returns
    -------
    heatmap          : 2D numpy array, shape (h, w), values in [0, 1]
    pred_index       : the class index used
    confidence       : prediction confidence for that class
    """
    # Build a sub-model with two outputs:
    #   (a) the activations of the last conv layer
    #   (b) the final predictions
    # First, find the last conv layer inside the base network
    grad_model = None

    # Try to find the last conv layer in the MobileNetV2 base
    for layer in model.layers:
        if 'mobilenetv2' in layer.name:
            # Look for the named layer inside the base
            try:
                target_layer = layer.get_layer(last_conv_layer_name)
                grad_model = keras.Model(
                    inputs=model.inputs,
                    outputs=[target_layer.output, model.output]
                )
            except ValueError:
                pass
            break

    # Fallback: search all layers in the top-level model
    if grad_model is None:
        for layer in model.layers:
            if layer.name == last_conv_layer_name:
                grad_model = keras.Model(
                    inputs=model.inputs,
                    outputs=[layer.output, model.output]
                )
                break

    if grad_model is None:
        # Build with the last convolutional layer we can find
        last_conv = None
        for layer in model.layers:
            if isinstance(layer, layers.Conv2D):
                last_conv = layer
        if last_conv is None and mn_base_layer is not None:
            for sub in mn_base_layer.layers:
                if isinstance(sub, layers.Conv2D):
                    last_conv = sub
        if last_conv is not None:
            grad_model = keras.Model(
                inputs=model.inputs,
                outputs=[last_conv.output, model.output]
            )
        else:
            print("  WARNING: could not build grad model; returning blank heatmap")
            return np.zeros((7, 7)), 0, 0.0

    # Forward pass inside GradientTape to record gradients
    with tf.GradientTape() as tape:
        # tape.watch ensures we track gradients through the conv output tensor
        conv_outputs, predictions = grad_model(img_array, training=False)
        tape.watch(conv_outputs)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])   # top predicted class
        # The scalar we differentiate: the score for the chosen class
        class_score = predictions[:, pred_index]

    # Gradient of class score w.r.t. conv layer output
    # Shape: same as conv_outputs — (1, h, w, num_filters)
    grads = tape.gradient(class_score, conv_outputs)

    # Global-average-pool the gradients over the spatial dimensions (h, w)
    # α_k = mean over all (i,j) of ∂y^c / ∂A^k_{ij}
    # Shape: (1, 1, 1, num_filters) → (num_filters,)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weighted combination of feature maps
    # conv_outputs[0] shape: (h, w, num_filters)
    conv_outputs_np = conv_outputs[0].numpy()
    pooled_grads_np = pooled_grads.numpy()

    # Multiply each feature map by its importance weight
    for k in range(pooled_grads_np.shape[0]):
        conv_outputs_np[:, :, k] *= pooled_grads_np[k]

    # Sum across all feature maps → (h, w)
    heatmap = np.mean(conv_outputs_np, axis=-1)

    # ReLU: keep only positive contributions
    # (negative values vote AGAINST the class; we ignore them)
    heatmap = np.maximum(heatmap, 0)

    # Normalise to [0, 1]
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

    confidence = float(predictions[0][pred_index])
    return heatmap, int(pred_index), confidence


def overlay_gradcam(original_img_uint8, heatmap, alpha=0.4):
    """
    Superimpose a Grad-CAM heatmap on the original image.

    Parameters
    ----------
    original_img_uint8 : uint8 image array (H, W, 3)
    heatmap            : 2D float array in [0, 1]
    alpha              : transparency of heatmap overlay (0=invisible, 1=opaque)

    Returns
    -------
    overlay : uint8 image (H, W, 3)
    """
    h, w = original_img_uint8.shape[:2]
    # Upsample heatmap to the original image size
    heatmap_resized = np.array(
        tf.image.resize(heatmap[..., np.newaxis], [h, w])
    )[:, :, 0]

    # Apply jet colourmap → RGBA (0–255)
    colormap  = cm.get_cmap('jet')
    heatmap_coloured = colormap(heatmap_resized)[:, :, :3]    # drop alpha channel
    heatmap_coloured = (heatmap_coloured * 255).astype(np.uint8)

    # Blend: (1-alpha)×original + alpha×heatmap
    overlay = ((1 - alpha) * original_img_uint8 + alpha * heatmap_coloured)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# Find the name of the last conv layer in the MobileNetV2 base
last_conv_name = None
if mn_base_layer is not None:
    for sub_layer in reversed(mn_base_layer.layers):
        if isinstance(sub_layer, layers.Conv2D):
            last_conv_name = sub_layer.name
            break
if last_conv_name is None:
    last_conv_name = 'Conv_1'   # fallback name in MobileNetV2

print(f"Using last conv layer: '{last_conv_name}' for Grad-CAM")

# Compute Grad-CAM for all visualisation images
print("\nComputing Grad-CAM heatmaps...")
heatmaps, pred_indices, confidences = [], [], []
for i in range(len(VIS_INDICES)):
    img_input = vis_images_proc[i:i+1]                       # (1, 96, 96, 3)
    hm, pi, conf = make_gradcam_heatmap(model, img_input, last_conv_name)
    heatmaps.append(hm)
    pred_indices.append(pi)
    confidences.append(conf)
    print(f"  Image {i}: true={CLASS_NAMES[vis_labels[i]]:<12} "
          f"pred={CLASS_NAMES[pi]:<12} conf={conf:.3f}")

# --- Visualise original + heatmap + overlay ---
fig, axes = plt.subplots(len(VIS_INDICES), 3, figsize=(12, 4 * len(VIS_INDICES)))
fig.suptitle('Grad-CAM Visualisations', fontsize=14, fontweight='bold')

for i in range(len(VIS_INDICES)):
    raw = vis_images_raw[i]       # original uint8 32×32 image
    hm  = heatmaps[i]
    # Upsample original for display
    raw_big = np.array(tf.image.resize(raw[np.newaxis], [IMG_SIZE, IMG_SIZE])[0],
                       dtype=np.uint8)
    overlay = overlay_gradcam(raw_big, hm, alpha=0.45)

    axes[i, 0].imshow(raw_big)
    axes[i, 0].set_title(f'Original\nTrue: {CLASS_NAMES[vis_labels[i]]}', fontsize=9)
    axes[i, 0].axis('off')

    axes[i, 1].imshow(hm, cmap='jet')
    axes[i, 1].set_title(f'Grad-CAM Heatmap\nPred: {CLASS_NAMES[pred_indices[i]]} '
                         f'({confidences[i]*100:.1f}%)', fontsize=9)
    axes[i, 1].axis('off')

    axes[i, 2].imshow(overlay)
    axes[i, 2].set_title('Overlay (50% blend)', fontsize=9)
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig('lesson_18_gradcam_heatmaps.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson_18_gradcam_heatmaps.png")


# ===================================================================
# SECTION 4: SALIENCY MAPS + MULTI-CLASS GRAD-CAM COMPARISON
# ===================================================================
print("\n" + "=" * 60)
print("SECTION 4: Saliency Maps and Multi-Class Grad-CAM Comparison")
print("=" * 60)

def compute_saliency_map(model, img_array, pred_index=None):
    """
    Compute a vanilla gradient saliency map.
    |∂y^c / ∂x_{ij}| — gradient of class score w.r.t. input pixels.
    Bright pixels had the most influence on the prediction.
    """
    img_tensor = tf.Variable(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        preds = model(img_tensor, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        score = preds[:, pred_index]

    # Gradient of score w.r.t. input image
    grads = tape.gradient(score, img_tensor)   # shape (1, H, W, 3)

    # Take absolute value and max across colour channels
    saliency = tf.reduce_max(tf.abs(grads), axis=-1)[0].numpy()   # (H, W)
    # Normalise to [0, 1]
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)
    return saliency, int(pred_index)

# Demonstrate on first 3 vis images
print("Computing saliency maps...")
fig, axes = plt.subplots(3, 4, figsize=(14, 10))
fig.suptitle('Saliency Map vs Grad-CAM Comparison', fontsize=14, fontweight='bold')
col_titles = ['Original', 'Saliency Map', 'Grad-CAM Heatmap', 'Grad-CAM Overlay']
for col, title in enumerate(col_titles):
    axes[0, col].set_title(title, fontsize=11, fontweight='bold')

for row in range(3):
    img_input = vis_images_proc[row:row+1]
    raw       = vis_images_raw[row]
    raw_big   = np.array(tf.image.resize(raw[np.newaxis], [IMG_SIZE, IMG_SIZE])[0],
                         dtype=np.uint8)

    sal_map, _ = compute_saliency_map(model, img_input)
    hm         = heatmaps[row]
    overlay    = overlay_gradcam(raw_big, hm)

    axes[row, 0].imshow(raw_big)
    axes[row, 0].set_ylabel(
        f'True: {CLASS_NAMES[vis_labels[row]]}\nPred: {CLASS_NAMES[pred_indices[row]]}',
        fontsize=8
    )
    axes[row, 0].axis('off')

    axes[row, 1].imshow(sal_map, cmap='hot')
    axes[row, 1].axis('off')

    axes[row, 2].imshow(hm, cmap='jet')
    axes[row, 2].axis('off')

    axes[row, 3].imshow(overlay)
    axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('lesson_18_saliency_vs_gradcam.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_18_saliency_vs_gradcam.png")

# --- Multi-class Grad-CAM: same image, different target classes ---
print("\nMulti-class Grad-CAM: showing heatmaps for 3 different class targets...")
img_idx   = 0
img_input = vis_images_proc[img_idx:img_idx+1]
raw_big   = np.array(
    tf.image.resize(vis_images_raw[img_idx][np.newaxis], [IMG_SIZE, IMG_SIZE])[0],
    dtype=np.uint8
)

# Get full prediction vector
preds_full = model.predict(img_input, verbose=0)[0]
sorted_idx = np.argsort(preds_full)[::-1]   # class indices sorted by confidence

target_classes = [sorted_idx[0], sorted_idx[1], sorted_idx[-1]]   # best, 2nd, worst

fig, axes = plt.subplots(1, 4, figsize=(15, 4))
fig.suptitle(
    f'Multi-Class Grad-CAM  |  True class: {CLASS_NAMES[vis_labels[img_idx]]}',
    fontsize=13, fontweight='bold'
)

axes[0].imshow(raw_big)
axes[0].set_title('Original Image', fontsize=10)
axes[0].axis('off')

for col, tc in enumerate(target_classes):
    hm_mc, _, conf_mc = make_gradcam_heatmap(model, img_input, last_conv_name,
                                             pred_index=tc)
    overlay_mc = overlay_gradcam(raw_big, hm_mc)
    axes[col + 1].imshow(overlay_mc)
    axes[col + 1].set_title(
        f'Target: {CLASS_NAMES[tc]}\nConf: {preds_full[tc]*100:.1f}%', fontsize=9
    )
    axes[col + 1].axis('off')

plt.tight_layout()
plt.savefig('lesson_18_multiclass_gradcam.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_18_multiclass_gradcam.png")

print("\nNote on interpretations:")
print("  - Heatmap for the correct class should cover the object region")
print("  - Heatmap for a wrong class highlights different, confusing regions")
print("  - Scattered or background heatmaps indicate spurious correlations")

print("\n" + "=" * 60)
print("LESSON 18 COMPLETE")
print("Key takeaway: Grad-CAM reveals what spatial regions of the input")
print("drive the model's predictions — essential for debugging and trust.")
print("=" * 60)
