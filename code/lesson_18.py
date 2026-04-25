"""
Lesson 18: Model Visualization & Interpretability (Grad-CAM)
Implements Grad-CAM from scratch and visualizes CNN feature maps.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("cv2 not available; using matplotlib colormap for Grad-CAM overlay.")

print("TensorFlow version:", tf.__version__)
print("=" * 60)

# === SECTION 1: Build and Train a Small CNN ===
print("\n=== SECTION 1: Build and Train a Small CNN ===")

# Load MNIST dataset (60k training, 10k test grayscale 28x28 images)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] and add channel dimension
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train[..., np.newaxis]   # shape: (60000, 28, 28, 1)
x_test  = x_test[..., np.newaxis]    # shape: (10000, 28, 28, 1)

print(f"Training data shape: {x_train.shape}, labels: {y_train.shape}")
print(f"Test data shape:     {x_test.shape},  labels: {y_test.shape}")

# Build a small CNN with two conv blocks
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, 3, activation='relu', padding='same', name='conv1'),  # First conv
    layers.MaxPooling2D(2, name='pool1'),
    layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2'),  # Second conv
    layers.MaxPooling2D(2, name='pool2'),
    layers.Flatten(),
    layers.Dense(10, activation='softmax', name='output')
], name='small_cnn')

model.summary()

# Compile and train for 2 quick epochs
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("\nTraining for 2 epochs...")
model.fit(x_train, y_train, validation_split=0.1,
          epochs=2, batch_size=256, verbose=1)

# Evaluate on test set
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest accuracy: {test_acc:.4f}")

# Get predictions on full test set
y_pred_prob = model.predict(x_test, verbose=0)
y_pred = np.argmax(y_pred_prob, axis=1)

# === SECTION 2: Visualize Intermediate Feature Maps ===
print("\n=== SECTION 2: Visualize Intermediate Feature Maps ===")

# Create a model that outputs activations from both conv layers
layer_outputs = [model.get_layer('conv1').output,
                 model.get_layer('conv2').output]
activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)

# Pick the first test sample for visualization
sample_img = x_test[0:1]  # shape (1, 28, 28, 1)
print(f"Visualizing feature maps for a '{y_test[0]}' digit.")

# Run sample through intermediate model to get activations
activations = activation_model.predict(sample_img, verbose=0)
conv1_act, conv2_act = activations  # (1, 28, 28, 32), (1, 14, 14, 64)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
fig.suptitle('Feature Maps: conv1 (top 2 rows) & conv2 (bottom 2 rows)', fontsize=12)

# Show first 16 filters from conv1
for i in range(16):
    row, col = divmod(i, 8)
    axes[row, col].imshow(conv1_act[0, :, :, i], cmap='viridis')
    axes[row, col].axis('off')
    axes[row, col].set_title(f'c1f{i}', fontsize=6)

# Show first 16 filters from conv2
for i in range(16):
    row, col = divmod(i, 8)
    axes[row + 2, col].imshow(conv2_act[0, :, :, i], cmap='viridis')
    axes[row + 2, col].axis('off')
    axes[row + 2, col].set_title(f'c2f{i}', fontsize=6)

plt.tight_layout()
plt.savefig('/tmp/feature_maps.png', dpi=100)
print("Saved feature maps to /tmp/feature_maps.png")

# === SECTION 3: Implement Grad-CAM from Scratch ===
print("\n=== SECTION 3: Implement Grad-CAM from Scratch ===")

def grad_cam(model, image, class_idx, last_conv_layer_name):
    """
    Compute Grad-CAM heatmap for a given image and class index.
    Uses tf.GradientTape to get gradients of class score w.r.t. last conv output.
    """
    # Build a sub-model that outputs [last conv feature maps, final predictions]
    grad_model = keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        inputs = tf.cast(image, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        # Score for the target class
        loss = predictions[:, class_idx]

    # Compute gradient of class score w.r.t. conv feature map outputs
    grads = tape.gradient(loss, conv_outputs)  # shape: (1, H, W, C)

    # Pool gradients over spatial dimensions to get per-channel importance weights
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # shape: (C,)

    # Weight each feature map channel by its importance
    conv_outputs = conv_outputs[0]                        # shape: (H, W, C)
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]  # (H, W, 1)
    heatmap = tf.squeeze(heatmap)                         # (H, W)

    # Apply ReLU to keep only positive influences
    heatmap = tf.nn.relu(heatmap)

    # Normalize heatmap to [0, 1]
    heatmap = heatmap.numpy()
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Resize heatmap to match original image size using zoom
    h, w = image.shape[1], image.shape[2]
    zoom_factors = (h / heatmap.shape[0], w / heatmap.shape[1])
    from scipy.ndimage import zoom as scipy_zoom
    heatmap_resized = scipy_zoom(heatmap, zoom_factors, order=1)

    return heatmap_resized

print("grad_cam() function defined.")
print("Testing Grad-CAM on one sample...")
test_heatmap = grad_cam(model, x_test[0:1], int(y_test[0]), 'conv2')
print(f"Heatmap shape: {test_heatmap.shape}, min={test_heatmap.min():.3f}, max={test_heatmap.max():.3f}")

# === SECTION 4: Generate and Overlay Grad-CAM Heatmaps ===
print("\n=== SECTION 4: Generate and Overlay Grad-CAM Heatmaps ===")

# Select 4 test images to visualize
sample_indices = [0, 1, 2, 3]

fig, axes = plt.subplots(4, 3, figsize=(10, 14))
fig.suptitle('Grad-CAM: Original | Heatmap | Overlay', fontsize=13)

for row, idx in enumerate(sample_indices):
    img = x_test[idx:idx+1]          # (1, 28, 28, 1)
    true_label = y_test[idx]
    pred_label = y_pred[idx]

    # Generate Grad-CAM heatmap for predicted class
    heatmap = grad_cam(model, img, pred_label, 'conv2')

    img_2d = img[0, :, :, 0]         # (28, 28) grayscale for display

    # Overlay: blend grayscale image with colormap heatmap
    overlay = plt.cm.jet(heatmap)[:, :, :3] * 0.5 + np.stack([img_2d]*3, axis=-1) * 0.5

    axes[row, 0].imshow(img_2d, cmap='gray')
    axes[row, 0].set_title(f'Original\ntrue={true_label}', fontsize=8)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(heatmap, cmap='jet')
    axes[row, 1].set_title(f'Heatmap\npred={pred_label}', fontsize=8)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title('Overlay', fontsize=8)
    axes[row, 2].axis('off')

plt.tight_layout()
plt.savefig('/tmp/gradcam.png', dpi=100)
print("Saved Grad-CAM visualizations to /tmp/gradcam.png")

# === SECTION 5: Compare Correct vs Incorrect Predictions ===
print("\n=== SECTION 5: Compare Correct vs Incorrect Predictions ===")

# Find indices of correct and incorrect predictions
correct_idx   = np.where(y_pred == y_test)[0][:2]    # First 2 correct
incorrect_idx = np.where(y_pred != y_test)[0][:2]    # First 2 incorrect

print(f"Correct predictions sample indices:   {correct_idx.tolist()}")
print(f"Incorrect predictions sample indices: {incorrect_idx.tolist()}")

fig, axes = plt.subplots(2, 6, figsize=(15, 6))
fig.suptitle('Grad-CAM: Correct (top) vs Incorrect (bottom) Predictions', fontsize=12)

for col_offset, (indices, label_str) in enumerate([(correct_idx, 'CORRECT'),
                                                    (incorrect_idx, 'WRONG')]):
    for j, idx in enumerate(indices):
        img = x_test[idx:idx+1]
        true_lbl = y_test[idx]
        pred_lbl = y_pred[idx]

        # Generate heatmap for predicted class
        heatmap = grad_cam(model, img, pred_lbl, 'conv2')
        img_2d  = img[0, :, :, 0]
        overlay = plt.cm.jet(heatmap)[:, :, :3] * 0.5 + np.stack([img_2d]*3, axis=-1) * 0.5

        base_col = col_offset * 3 + j * (0 if len(indices) == 1 else 3)
        # Three columns per example: original, heatmap, overlay
        plot_col = col_offset * 3 + j

        axes[0, plot_col].imshow(img_2d, cmap='gray')
        axes[0, plot_col].set_title(f'{label_str}\nTrue:{true_lbl} Pred:{pred_lbl}', fontsize=7)
        axes[0, plot_col].axis('off')

        axes[1, plot_col].imshow(overlay)
        axes[1, plot_col].set_title('Grad-CAM Overlay', fontsize=7)
        axes[1, plot_col].axis('off')

# Hide unused axes
for c in range(4, 6):
    axes[0, c].axis('off')
    axes[1, c].axis('off')

plt.tight_layout()
plt.savefig('/tmp/gradcam_comparison.png', dpi=100)
print("Saved comparison to /tmp/gradcam_comparison.png")
print("\nAnalysis: Grad-CAM highlights the regions that influenced the prediction.")
print("Correct predictions show focused activation on the digit strokes.")
print("Incorrect predictions may show diffuse or misleading activations.")

plt.show()
print("\nLesson 18 complete!")
