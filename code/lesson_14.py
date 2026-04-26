"""
Lesson 14: Data Augmentation
==============================
Demonstrates individual augmentations with PIL, ImageDataGenerator,
Keras augmentation layers, model training with/without augmentation,
and test-time augmentation (TTA).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image, ImageEnhance, ImageFilter
import io

print("=" * 60)
print("LESSON 14: Data Augmentation")
print("=" * 60)

# ── Load CIFAR-10 ─────────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test  = y_test.flatten()

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Pick a sample image for augmentation gallery (a cat)
sample_idx  = np.where(y_train == 3)[0][0]  # first cat in training set
sample_img  = X_train[sample_idx]            # shape (32, 32, 3), float32 [0,1]

def arr_to_pil(arr):
    """Convert a float32 numpy array [0,1] to a PIL Image."""
    return Image.fromarray((arr * 255).astype(np.uint8))

def pil_to_arr(img):
    """Convert PIL Image back to float32 numpy array [0,1]."""
    return np.array(img).astype('float32') / 255.0

# ──────────────────────────────────────────────────────────────
# SECTION 1: Individual Augmentations with PIL
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Individual Augmentations with PIL ---")

np.random.seed(42)
pil_img = arr_to_pil(sample_img)

# Define individual transforms
def horizontal_flip(img):
    """Mirror the image left-to-right."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def random_rotation(img, max_angle=30):
    """Rotate by a random angle within ±max_angle degrees."""
    angle = np.random.uniform(-max_angle, max_angle)
    return img.rotate(angle, fillcolor=(128, 128, 128))  # grey fill for border

def random_crop(img, crop_frac=0.85):
    """Randomly crop a fraction of the image and resize back."""
    w, h = img.size
    new_w = int(w * crop_frac)
    new_h = int(h * crop_frac)
    left  = np.random.randint(0, w - new_w + 1)
    top   = np.random.randint(0, h - new_h + 1)
    cropped = img.crop((left, top, left + new_w, top + new_h))
    return cropped.resize((w, h), Image.BILINEAR)

def color_jitter(img, brightness=0.4, contrast=0.4, saturation=0.3):
    """Randomly perturb brightness, contrast, and saturation."""
    # Brightness
    factor = 1 + np.random.uniform(-brightness, brightness)
    img = ImageEnhance.Brightness(img).enhance(factor)
    # Contrast
    factor = 1 + np.random.uniform(-contrast, contrast)
    img = ImageEnhance.Contrast(img).enhance(factor)
    # Saturation (Color)
    factor = 1 + np.random.uniform(-saturation, saturation)
    img = ImageEnhance.Color(img).enhance(factor)
    return img

def random_zoom(img, zoom_range=(0.8, 1.2)):
    """Zoom in or out within the given range."""
    w, h = img.size
    zoom_factor = np.random.uniform(*zoom_range)
    new_w = int(w * zoom_factor)
    new_h = int(h * zoom_factor)
    if zoom_factor > 1:                          # zoom in: crop the center
        left = (new_w - w) // 2
        top  = (new_h - h) // 2
        img  = img.resize((new_w, new_h), Image.BILINEAR)
        img  = img.crop((left, top, left + w, top + h))
    else:                                         # zoom out: paste on grey background
        bg = Image.new('RGB', (w, h), (128, 128, 128))
        img = img.resize((new_w, new_h), Image.BILINEAR)
        paste_x = (w - new_w) // 2
        paste_y = (h - new_h) // 2
        bg.paste(img, (paste_x, paste_y))
        img = bg
    return img

def cutout(img, frac=0.3):
    """Mask a random rectangular region with grey pixels."""
    arr = np.array(img).copy()
    h, w = arr.shape[:2]
    cut_h = int(h * frac)
    cut_w = int(w * frac)
    top   = np.random.randint(0, h - cut_h + 1)
    left  = np.random.randint(0, w - cut_w + 1)
    arr[top:top + cut_h, left:left + cut_w] = 128   # grey fill
    return Image.fromarray(arr)

# Build the augmentation gallery
augments = {
    'Original':            pil_img,
    'Horizontal Flip':     horizontal_flip(pil_img),
    'Rotation (±30°)':     random_rotation(pil_img, 30),
    'Random Crop':         random_crop(pil_img, 0.85),
    'Color Jitter':        color_jitter(pil_img),
    'Zoom In/Out':         random_zoom(pil_img),
    'Cutout':              cutout(pil_img, 0.35),
}

print(f"Augmentations applied to sample class: {CLASS_NAMES[y_train[sample_idx]]}")
for name in augments:
    print(f"  ✓ {name}")

fig, axes = plt.subplots(1, len(augments), figsize=(3 * len(augments), 3))
for ax, (name, img) in zip(axes, augments.items()):
    ax.imshow(img)
    ax.set_title(name, fontsize=9)
    ax.axis('off')

plt.suptitle('Individual Augmentation Techniques (PIL)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('lesson14_augmentation_gallery.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_augmentation_gallery.png")

# ──────────────────────────────────────────────────────────────
# SECTION 2: ImageDataGenerator — Visualise Augmented Batches
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: ImageDataGenerator Augmented Batches ---")

# ImageDataGenerator expects [0, 255] uint8 for display; it normalises internally
datagen = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=[0.7, 1.3],
    fill_mode='nearest',
)

# Generate 8 augmented versions of the same image
img_uint8 = (sample_img * 255).astype(np.uint8)           # (32, 32, 3)
img_batch = img_uint8[np.newaxis, ...]                     # add batch dim → (1, 32, 32, 3)

print("Generating 8 augmented versions using ImageDataGenerator...")
fig, axes = plt.subplots(2, 5, figsize=(13, 5))
axes = axes.ravel()
axes[0].imshow(sample_img)
axes[0].set_title('Original', fontsize=9)
axes[0].axis('off')

for i, aug_batch in enumerate(datagen.flow(img_batch, batch_size=1)):
    if i >= 8:
        break
    aug_img = aug_batch[0].astype(np.uint8)
    axes[i + 1].imshow(aug_img)
    axes[i + 1].set_title(f'Augmented #{i+1}', fontsize=9)
    axes[i + 1].axis('off')

# Hide unused axes
for j in range(i + 2, len(axes)):
    axes[j].axis('off')

plt.suptitle('ImageDataGenerator: Augmented Versions of One Image', fontsize=12)
plt.tight_layout()
plt.savefig('lesson14_imagedatagenerator.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_imagedatagenerator.png")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Keras Augmentation Layers Pipeline
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Keras Augmentation Layers Pipeline ---")

# These layers are inactive during inference (training=False)
augmentation_pipeline = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(factor=0.1),         # ±10% of 2π
    keras.layers.RandomZoom(height_factor=0.1),
    keras.layers.RandomContrast(factor=0.2),
    keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
], name='augmentation')

print("Keras augmentation pipeline layers:")
augmentation_pipeline.summary()

# Apply pipeline to a batch of images and visualise
sample_batch = X_train[:8]   # (8, 32, 32, 3)
# Apply augmentation in training mode
aug_batch_tf = augmentation_pipeline(sample_batch, training=True).numpy()
aug_batch_infer = augmentation_pipeline(sample_batch, training=False).numpy()

fig, axes = plt.subplots(3, 8, figsize=(16, 6))
for i in range(8):
    axes[0, i].imshow(np.clip(sample_batch[i], 0, 1))
    axes[0, i].set_title(CLASS_NAMES[y_train[i]], fontsize=8)
    axes[0, i].axis('off')

    axes[1, i].imshow(np.clip(aug_batch_tf[i], 0, 1))
    axes[1, i].set_title('Augmented', fontsize=8)
    axes[1, i].axis('off')

    axes[2, i].imshow(np.clip(aug_batch_infer[i], 0, 1))
    axes[2, i].set_title('Inference\n(no aug)', fontsize=8)
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('Original', fontsize=9, rotation=0, labelpad=55)
axes[1, 0].set_ylabel('Training\n(augmented)', fontsize=9, rotation=0, labelpad=55)
axes[2, 0].set_ylabel('Inference\n(identity)', fontsize=9, rotation=0, labelpad=55)

plt.suptitle('Keras Layers Augmentation: Train vs Inference Mode', fontsize=12)
plt.tight_layout()
plt.savefig('lesson14_keras_augmentation.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_keras_augmentation.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: Train With vs Without Augmentation
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: Training With vs Without Augmentation ---")

N_TRAIN = 10000
X_tr  = X_train[:N_TRAIN]
y_tr  = y_train[:N_TRAIN]
X_val = X_train[N_TRAIN:N_TRAIN + 3000]
y_val = y_train[N_TRAIN:N_TRAIN + 3000]
EPOCHS = 25
BATCH  = 64

def build_cnn(with_augmentation=False):
    """CNN with optional augmentation as first layers."""
    layers = []
    if with_augmentation:
        layers += [
            keras.layers.RandomFlip("horizontal", input_shape=(32, 32, 3)),
            keras.layers.RandomRotation(0.1),
            keras.layers.RandomZoom(0.1),
        ]
        layers.append(keras.layers.Conv2D(64, 3, padding='same', activation='relu'))
    else:
        layers.append(keras.layers.Conv2D(64, 3, padding='same', activation='relu',
                                          input_shape=(32, 32, 3)))
    layers += [
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax'),
    ]
    return keras.Sequential(layers)

print("  Training WITHOUT augmentation...")
m_no_aug = build_cnn(with_augmentation=False)
m_no_aug.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])
hist_no_aug = m_no_aug.fit(X_tr, y_tr,
                            validation_data=(X_val, y_val),
                            epochs=EPOCHS, batch_size=BATCH, verbose=0)
print(f"    Final val accuracy (no aug): {hist_no_aug.history['val_accuracy'][-1]:.4f}")

print("  Training WITH augmentation (Keras layers)...")
m_aug = build_cnn(with_augmentation=True)
m_aug.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
hist_aug = m_aug.fit(X_tr, y_tr,
                     validation_data=(X_val, y_val),
                     epochs=EPOCHS, batch_size=BATCH, verbose=0)
print(f"    Final val accuracy (with aug): {hist_aug.history['val_accuracy'][-1]:.4f}")

# Plot comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(hist_no_aug.history['accuracy'],     'b-', linewidth=2, label='No Aug - Train')
axes[0].plot(hist_no_aug.history['val_accuracy'], 'b--', linewidth=2, label='No Aug - Val')
axes[0].plot(hist_aug.history['accuracy'],        'r-', linewidth=2, label='Aug - Train')
axes[0].plot(hist_aug.history['val_accuracy'],    'r--', linewidth=2, label='Aug - Val')
axes[0].set_title('Accuracy: With vs Without Augmentation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(hist_no_aug.history['val_loss'], 'b-', linewidth=2, label='No Augmentation')
axes[1].plot(hist_aug.history['val_loss'],    'r-', linewidth=2, label='With Augmentation')
axes[1].set_title('Validation Loss: With vs Without Augmentation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Val Loss')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lesson14_aug_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_aug_comparison.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: Test-Time Augmentation (TTA)
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Test-Time Augmentation (TTA) ---")

# Use the augmented model from Section 4
tta_augment = keras.Sequential([
    keras.layers.RandomFlip("horizontal"),
    keras.layers.RandomRotation(factor=0.05),
    keras.layers.RandomZoom(height_factor=0.05),
])

def predict_with_tta(model, images, n_augments=8):
    """
    Predict with TTA by averaging predictions over n_augments augmented copies.
    
    Args:
        model: trained Keras model
        images: numpy array of images (N, H, W, C)
        n_augments: number of augmented versions to average over
    Returns:
        averaged probability array (N, n_classes)
    """
    preds = []
    for _ in range(n_augments):
        # Apply random augmentation in training mode, then get predictions
        aug_images = tta_augment(images, training=True)
        pred = model(aug_images, training=False)     # inference mode for BN/Dropout
        preds.append(pred.numpy())
    return np.mean(preds, axis=0)  # average probabilities across all augmented versions

# Evaluate on a subset of test images for speed
N_EVAL = 1000
X_eval = X_test[:N_EVAL]
y_eval = y_test[:N_EVAL]

# Standard (no TTA) prediction
y_pred_standard = m_aug.predict(X_eval, verbose=0).argmax(axis=1)
acc_standard = np.mean(y_pred_standard == y_eval)

# TTA prediction
print(f"  Running TTA with 8 augmented copies per image ({N_EVAL} test images)...")
y_pred_tta_probs = predict_with_tta(m_aug, X_eval, n_augments=8)
y_pred_tta = y_pred_tta_probs.argmax(axis=1)
acc_tta = np.mean(y_pred_tta == y_eval)

print(f"\n  Standard prediction accuracy:  {acc_standard:.4f}")
print(f"  TTA (×8) prediction accuracy:  {acc_tta:.4f}")
print(f"  TTA improvement:               {acc_tta - acc_standard:+.4f}")

# Visualise TTA predictions on a few individual examples
fig, axes = plt.subplots(2, 8, figsize=(16, 5))
for i in range(8):
    # Original image
    axes[0, i].imshow(X_eval[i])
    true_class = CLASS_NAMES[y_eval[i]]
    pred_class = CLASS_NAMES[y_pred_standard[i]]
    color = 'green' if y_pred_standard[i] == y_eval[i] else 'red'
    axes[0, i].set_title(f'True: {true_class}\nPred: {pred_class}', fontsize=7, color=color)
    axes[0, i].axis('off')

    # Confidence bar chart (standard vs TTA)
    classes_range = np.arange(10)
    standard_conf = m_aug.predict(X_eval[i:i+1], verbose=0)[0]
    tta_conf      = y_pred_tta_probs[i]
    axes[1, i].bar(classes_range - 0.2, standard_conf, 0.4, label='Standard', color='blue', alpha=0.7)
    axes[1, i].bar(classes_range + 0.2, tta_conf,      0.4, label='TTA',      color='orange', alpha=0.7)
    axes[1, i].set_xticks([])
    axes[1, i].set_ylim(0, 1)
    if i == 0:
        axes[1, i].legend(fontsize=6)

plt.suptitle('TTA: Standard vs Averaged Predictions (Blue=Standard, Orange=TTA)', fontsize=11)
plt.tight_layout()
plt.savefig('lesson14_tta.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_tta.png")

# TTA accuracy vs number of augmented copies
print("\n  TTA accuracy vs number of augmented copies:")
tta_accs = []
n_aug_values = [1, 2, 4, 8, 16]
for n in n_aug_values:
    probs = predict_with_tta(m_aug, X_eval, n_augments=n)
    acc   = np.mean(probs.argmax(axis=1) == y_eval)
    tta_accs.append(acc)
    print(f"    n_augments = {n:3d}: accuracy = {acc:.4f}")

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(n_aug_values, tta_accs, 'bo-', markersize=8, linewidth=2)
ax.axhline(acc_standard, color='r', linestyle='--', label=f'Standard (no TTA) = {acc_standard:.4f}')
ax.set_xlabel('Number of TTA Augmented Copies')
ax.set_ylabel('Accuracy')
ax.set_title('TTA Accuracy vs Number of Augmented Copies')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lesson14_tta_scaling.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson14_tta_scaling.png")

print("\n" + "=" * 60)
print("Lesson 14 complete. Outputs saved:")
print("  lesson14_augmentation_gallery.png")
print("  lesson14_imagedatagenerator.png")
print("  lesson14_keras_augmentation.png")
print("  lesson14_aug_comparison.png")
print("  lesson14_tta.png")
print("  lesson14_tta_scaling.png")
print("=" * 60)
