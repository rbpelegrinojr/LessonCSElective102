"""
Lesson 20: Capstone Project — End-to-End Image Classifier
A complete pipeline combining all 20 lessons: EDA, data pipeline, transfer
learning (feature extraction + fine-tuning), evaluation, Grad-CAM, and deployment.
Uses CIFAR-10 (10 classes, always available via Keras).
"""

import matplotlib
matplotlib.use('Agg')

# Standard scientific stack
import numpy as np
import matplotlib.pyplot as plt
import os
import time

# Deep learning
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Evaluation
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print("TensorFlow:", tf.__version__)
print("NumPy:", np.__version__)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# ============================================================
# === SECTION 1: Exploratory Data Analysis (EDA) ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 1: Exploratory Data Analysis (EDA)")
print("=" * 60)

(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

# Flatten label arrays from (N,1) to (N,)
y_train_full = y_train_full.flatten()
y_test       = y_test.flatten()

print(f"Training images : {x_train_full.shape}  dtype={x_train_full.dtype}")
print(f"Test images     : {x_test.shape}  dtype={x_test.dtype}")
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")
print(f"Pixel range: [{x_train_full.min()}, {x_train_full.max()}]")
print(f"Mean pixel value: {x_train_full.mean():.2f}  Std: {x_train_full.std():.2f}")

# Plot 5 examples per class (5 × 10 grid)
fig, axes = plt.subplots(5, 10, figsize=(16, 8))
fig.suptitle('CIFAR-10 — 5 Examples per Class', fontsize=14)
for col, cls in enumerate(range(10)):
    idx = np.where(y_train_full == cls)[0][:5]
    for row, i in enumerate(idx):
        axes[row, col].imshow(x_train_full[i])
        axes[row, col].axis('off')
        if row == 0:
            axes[row, col].set_title(CLASS_NAMES[cls], fontsize=8)
plt.tight_layout()
plt.savefig('/tmp/capstone_eda.png', dpi=100)
print("Saved /tmp/capstone_eda.png")

# Class distribution bar chart
fig2, ax2 = plt.subplots(figsize=(10, 4))
unique, counts = np.unique(y_train_full, return_counts=True)
ax2.bar([CLASS_NAMES[u] for u in unique], counts, color='steelblue')
ax2.set_title('CIFAR-10 Training Set — Class Distribution')
ax2.set_ylabel('Number of samples')
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('/tmp/capstone_class_dist.png', dpi=100)
print("Saved /tmp/capstone_class_dist.png")

# ============================================================
# === SECTION 2: Data Pipeline with Augmentation ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 2: Data Pipeline with Augmentation")
print("=" * 60)

# Use a subset to keep training fast in demo
N_TRAIN   = 5000
N_VAL     = 1000
N_TEST    = 1000
IMG_SIZE  = 96   # MobileNetV2 minimum size
BATCH     = 32

# Subset
x_tr = x_train_full[:N_TRAIN].astype('float32') / 255.0
y_tr = y_train_full[:N_TRAIN]
x_va = x_train_full[N_TRAIN:N_TRAIN + N_VAL].astype('float32') / 255.0
y_va = y_train_full[N_TRAIN:N_TRAIN + N_VAL]
x_te = x_test[:N_TEST].astype('float32') / 255.0
y_te = y_test[:N_TEST]

print(f"Train: {x_tr.shape}, Val: {x_va.shape}, Test: {x_te.shape}")

# Resize to 96×96 using tf.image
def resize_batch(x, size=IMG_SIZE):
    return tf.image.resize(x, [size, size]).numpy()

print(f"Resizing images to {IMG_SIZE}×{IMG_SIZE} for MobileNetV2...")
x_tr = resize_batch(x_tr)
x_va = resize_batch(x_va)
x_te = resize_batch(x_te)
print(f"After resize — Train: {x_tr.shape}")

# Augmentation layer (applied only during training)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
], name="augmentation")

# Build tf.data pipelines
def make_dataset(x, y, augment=False, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), seed=42)
    if augment:
        ds = ds.map(lambda img, lbl: (data_augmentation(img, training=True), lbl),
                    num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(x_tr, y_tr, augment=True,  shuffle=True)
val_ds   = make_dataset(x_va, y_va, augment=False, shuffle=False)
test_ds  = make_dataset(x_te, y_te, augment=False, shuffle=False)
print("tf.data pipelines ready (train augmented, val/test not augmented)")

# ============================================================
# === SECTION 3: Transfer Learning — Feature Extraction Phase ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 3: Transfer Learning — Feature Extraction Phase")
print("=" * 60)

# Load MobileNetV2 without the top classification layer
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze ALL base model layers
base_model.trainable = False
print(f"Base model layers: {len(base_model.layers)}, all frozen")

# Build full classification model
inputs  = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x       = base_model(inputs, training=False)          # frozen: BN in inference mode
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.3)(x)
outputs = layers.Dense(10, activation='softmax')(x)
model   = models.Model(inputs, outputs, name='capstone_model')

total_params     = model.count_params()
trainable_params = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"Total params:     {total_params:,}")
print(f"Trainable params: {trainable_params:,}  ({100*trainable_params/total_params:.1f}%)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("\nPhase 1 — training classification head only (5 epochs max)...")
hist1 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=5, callbacks=[early_stop], verbose=1
)

# Plot phase-1 curves
fig3, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4))
epochs1 = range(1, len(hist1.history['loss']) + 1)
ax_l.plot(epochs1, hist1.history['loss'],     label='Train loss')
ax_l.plot(epochs1, hist1.history['val_loss'], label='Val loss')
ax_l.set_title('Phase 1 — Loss')
ax_l.set_xlabel('Epoch'); ax_l.legend()
ax_a.plot(epochs1, hist1.history['accuracy'],     label='Train acc')
ax_a.plot(epochs1, hist1.history['val_accuracy'], label='Val acc')
ax_a.set_title('Phase 1 — Accuracy')
ax_a.set_xlabel('Epoch'); ax_a.legend()
plt.tight_layout()
plt.savefig('/tmp/capstone_phase1.png', dpi=100)
print("Saved /tmp/capstone_phase1.png")
val_acc1 = max(hist1.history['val_accuracy'])
print(f"Best Phase-1 val accuracy: {val_acc1*100:.2f}%")

# ============================================================
# === SECTION 4: Fine-Tuning Phase + Full Evaluation ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 4: Fine-Tuning Phase + Full Evaluation")
print("=" * 60)

# Unfreeze the last 30 layers of the base model
base_model.trainable = True
FREEZE_UP_TO = len(base_model.layers) - 30
for layer in base_model.layers[:FREEZE_UP_TO]:
    layer.trainable = False

trainable_now = sum(tf.size(v).numpy() for v in model.trainable_variables)
print(f"Unfroze last 30 base layers.  Trainable params now: {trainable_now:,}")

# Re-compile with a much smaller learning rate
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Phase 2 — fine-tuning (5 epochs max, lr=1e-5)...")
hist2 = model.fit(
    train_ds, validation_data=val_ds,
    epochs=5, callbacks=[early_stop], verbose=1
)

# Combined training curve
epochs2_offset = len(hist1.history['loss'])
epochs2 = range(epochs2_offset + 1, epochs2_offset + len(hist2.history['loss']) + 1)
fig4, (ax_l2, ax_a2) = plt.subplots(1, 2, figsize=(12, 4))
all_epochs = list(range(1, epochs2_offset + len(hist2.history['loss']) + 1))
all_loss   = hist1.history['loss'] + hist2.history['loss']
all_vloss  = hist1.history['val_loss'] + hist2.history['val_loss']
all_acc    = hist1.history['accuracy'] + hist2.history['accuracy']
all_vacc   = hist1.history['val_accuracy'] + hist2.history['val_accuracy']
ax_l2.plot(all_epochs, all_loss,  label='Train loss')
ax_l2.plot(all_epochs, all_vloss, label='Val loss')
ax_l2.axvline(epochs2_offset + 0.5, linestyle='--', color='gray', label='Phase boundary')
ax_l2.set_title('Combined Loss (Phase 1 + Fine-Tune)')
ax_l2.set_xlabel('Epoch'); ax_l2.legend()
ax_a2.plot(all_epochs, all_acc,  label='Train acc')
ax_a2.plot(all_epochs, all_vacc, label='Val acc')
ax_a2.axvline(epochs2_offset + 0.5, linestyle='--', color='gray', label='Phase boundary')
ax_a2.set_title('Combined Accuracy')
ax_a2.set_xlabel('Epoch'); ax_a2.legend()
plt.tight_layout()
plt.savefig('/tmp/capstone_combined.png', dpi=100)
print("Saved /tmp/capstone_combined.png")

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_ds, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

# Confusion matrix
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
cm = confusion_matrix(y_te, y_pred)

fig5, ax5 = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax5)
ax5.set_xlabel('Predicted'); ax5.set_ylabel('True')
ax5.set_title('Confusion Matrix — CIFAR-10 Capstone Model')
plt.tight_layout()
plt.savefig('/tmp/capstone_confusion.png', dpi=100)
print("Saved /tmp/capstone_confusion.png")

print("\nClassification Report:")
print(classification_report(y_te, y_pred, target_names=CLASS_NAMES))

# ============================================================
# === SECTION 5: Grad-CAM Visualization + Save Final Model ===
# ============================================================
print("\n" + "=" * 60)
print("SECTION 5: Grad-CAM Visualization + Save Final Model")
print("=" * 60)

# ---- Grad-CAM helper ----
def get_grad_cam(model, image, class_idx):
    """
    Compute Grad-CAM heatmap for a single image.
    Returns heatmap resized to input image spatial dims.
    """
    # Find the last Conv layer in the base model
    last_conv_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model):          # the base_model sub-model
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    last_conv_name = sub.name
                    break
        if last_conv_name:
            break
    if last_conv_name is None:
        # Fallback: use any Conv2D at the top level
        for layer in reversed(model.layers):
            if isinstance(layer, layers.Conv2D):
                last_conv_name = layer.name
                break

    # Build a sub-model that outputs (conv_output, final_predictions)
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        img_tensor = tf.cast(image[np.newaxis, ...], tf.float32)
        conv_outputs, predictions = grad_model(img_tensor)
        loss = predictions[:, class_idx]

    # Gradients of the class score w.r.t. conv output
    grads = tape.gradient(loss, conv_outputs)[0]         # (H, W, C)
    # Global-average-pool the gradients to get channel weights
    weights = tf.reduce_mean(grads, axis=(0, 1))         # (C,)
    # Weighted sum of feature maps
    cam = tf.reduce_sum(conv_outputs[0] * weights, axis=-1)  # (H, W)
    # ReLU: we only care about positive influence
    cam = tf.nn.relu(cam).numpy()
    # Normalise
    if cam.max() > 0:
        cam = cam / cam.max()
    # Resize to input spatial dimensions
    cam_resized = tf.image.resize(cam[..., np.newaxis], [IMG_SIZE, IMG_SIZE]).numpy()[..., 0]
    return cam_resized


# Find correct and incorrect examples
correct_idx   = np.where(y_pred == y_te)[0][:5]
incorrect_idx = np.where(y_pred != y_te)[0][:5]
n_show = min(len(correct_idx), len(incorrect_idx), 3)

fig6, axes6 = plt.subplots(2, n_show * 3, figsize=(n_show * 9, 6))
fig6.suptitle('Grad-CAM: Correct (top) vs Incorrect (bottom) Predictions', fontsize=12)

for row, indices in enumerate([correct_idx[:n_show], incorrect_idx[:n_show]]):
    for col, idx in enumerate(indices):
        img   = x_te[idx]
        true  = CLASS_NAMES[y_te[idx]]
        pred  = CLASS_NAMES[y_pred[idx]]
        heatmap = get_grad_cam(model, img, y_pred[idx])

        # Original image
        ax = axes6[row, col * 3]
        ax.imshow(img)
        ax.set_title(f"True: {true}\nPred: {pred}", fontsize=7)
        ax.axis('off')

        # Heatmap
        ax2 = axes6[row, col * 3 + 1]
        ax2.imshow(heatmap, cmap='jet')
        ax2.set_title("Grad-CAM", fontsize=7)
        ax2.axis('off')

        # Overlay
        ax3 = axes6[row, col * 3 + 2]
        ax3.imshow(img)
        ax3.imshow(heatmap, cmap='jet', alpha=0.45)
        ax3.set_title("Overlay", fontsize=7)
        ax3.axis('off')

plt.tight_layout()
plt.savefig('/tmp/capstone_gradcam.png', dpi=100)
print("Saved /tmp/capstone_gradcam.png")

# Save final model
model_path = '/tmp/capstone_final_model.h5'
model.save(model_path)
model_size = os.path.getsize(model_path) / (1024 * 1024)
print(f"\nFinal model saved: {model_path}  ({model_size:.1f} MB)")


# Production inference function
def predict_image(model, image, class_names=CLASS_NAMES, input_size=(IMG_SIZE, IMG_SIZE)):
    """Run full preprocessing → inference → formatted output for a single image."""
    img = np.array(image, dtype=np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    if img.shape[:2] != tuple(input_size):
        img = tf.image.resize(img, input_size).numpy()
    probs      = model.predict(img[np.newaxis, ...], verbose=0)[0]
    pred_idx   = int(np.argmax(probs))
    confidence = float(probs[pred_idx])
    top3 = [(class_names[i], float(probs[i]))
            for i in np.argsort(probs)[::-1][:3]]
    return {'class': class_names[pred_idx], 'confidence': confidence, 'top3': top3}


print("\nSample inferences with predict_image():")
for i in range(5):
    result = predict_image(model, x_te[i])
    true_label = CLASS_NAMES[y_te[i]]
    mark = "✓" if result['class'] == true_label else "✗"
    print(f"  {mark} True={true_label:12s}  Pred={result['class']:12s}  "
          f"Conf={result['confidence']*100:.1f}%")

# ---- Course completion summary ----
LESSON_TITLES = [
    "What Is Image Classification?",
    "Understanding Digital Images & Pixels",
    "Introduction to Neural Networks",
    "From Dense Networks to CNNs",
    "The Convolution Operation Deep Dive",
    "Activation Functions",
    "Pooling Layers",
    "Building Your First Complete CNN",
    "Dataset Preparation & Loading",
    "Training a CNN",
    "Loss Functions & Optimizers",
    "Evaluating Your Model — Metrics & Confusion Matrix",
    "Overfitting & Regularization Techniques",
    "Data Augmentation",
    "Batch Normalization & Dropout",
    "Transfer Learning",
    "Fine-Tuning Pretrained Models",
    "Model Visualization & Interpretability (Grad-CAM)",
    "Deploying Your CNN Model",
    "Capstone Project — End-to-End Image Classifier",
]

print("\n" + "=" * 60)
print("COURSE COMPLETE: Image Classification Using CNNs")
print("=" * 60)
print("Lessons covered:")
for i, title in enumerate(LESSON_TITLES, 1):
    print(f"  Lesson {i:02d}: {title}")
print("=" * 60)
print("Congratulations! You have completed CS Elective 102.")
print("=" * 60)

plt.show()
