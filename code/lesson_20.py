"""
Lesson 20: Capstone Project — End-to-End Image Classifier
===========================================================
Complete end-to-end pipeline on CIFAR-10:
  - Exploratory Data Analysis (EDA)
  - Data augmentation pipeline
  - Transfer learning with MobileNetV2 (feature extraction + fine-tuning)
  - Full evaluation: accuracy, per-class F1, confusion matrix
  - Grad-CAM visualisation on correct and misclassified examples
  - Error analysis
  - Model save + TFLite conversion
  - Comprehensive model report
"""

import os, time, warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess

tf.random.set_seed(42)
np.random.seed(42)

print("=" * 65)
print("LESSON 20: Capstone Project — End-to-End Image Classifier")
print("=" * 65)
print(f"TensorFlow version: {tf.__version__}")

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']
NUM_CLASSES = 10
IMG_SIZE    = 96
BATCH_SIZE  = 64

# ===================================================================
# SECTION 1: EXPLORATORY DATA ANALYSIS
# ===================================================================
print("\n" + "=" * 65)
print("SECTION 1: Exploratory Data Analysis")
print("=" * 65)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_tr_flat = y_train.squeeze()
y_te_flat = y_test.squeeze()

print(f"Training set:  {x_train.shape}  dtype={x_train.dtype}")
print(f"Test set:      {x_test.shape}")
print(f"Pixel range:   [{x_train.min()}, {x_train.max()}]")

# Class distribution
counts_train = np.bincount(y_tr_flat)
counts_test  = np.bincount(y_te_flat)
print("\nClass distribution (train / test):")
for i, name in enumerate(CLASS_NAMES):
    print(f"  {i} {name:<12}: train={counts_train[i]:5d}  test={counts_test[i]:4d}")

# Per-channel mean and std
mean_per_channel = x_train.mean(axis=(0, 1, 2))
std_per_channel  = x_train.std(axis=(0, 1, 2))
print(f"\nPer-channel pixel mean (R,G,B): {mean_per_channel.astype(int)}")
print(f"Per-channel pixel std  (R,G,B): {std_per_channel.astype(int)}")

# EDA figure
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Lesson 20 Capstone — EDA", fontsize=15, fontweight='bold')

# Sample images grid
for i in range(20):
    ax = fig.add_subplot(4, 10, i + 1)
    ax.imshow(x_train[i])
    ax.set_title(CLASS_NAMES[y_tr_flat[i]], fontsize=7)
    ax.axis('off')

# Class distribution bar
ax_dist = fig.add_subplot(4, 2, 9)
ax_dist.bar(CLASS_NAMES, counts_train, color='steelblue', edgecolor='black')
ax_dist.set_title('Training Class Distribution')
ax_dist.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=8)
ax_dist.set_ylabel('Count')
ax_dist.grid(axis='y', alpha=0.3)

# Pixel brightness histogram
ax_hist = fig.add_subplot(4, 2, 10)
ax_hist.hist(x_train[..., 0].ravel(), bins=64, alpha=0.6, color='red',   label='R', density=True)
ax_hist.hist(x_train[..., 1].ravel(), bins=64, alpha=0.6, color='green', label='G', density=True)
ax_hist.hist(x_train[..., 2].ravel(), bins=64, alpha=0.6, color='blue',  label='B', density=True)
ax_hist.set_title('Pixel Intensity Distribution')
ax_hist.set_xlabel('Pixel Value'); ax_hist.set_ylabel('Density')
ax_hist.legend(fontsize=8)
ax_hist.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('lesson_20_eda.png', dpi=100, bbox_inches='tight')
plt.close()
print("\nSaved: lesson_20_eda.png")

# ===================================================================
# SECTION 2: DATA PIPELINE WITH AUGMENTATION
# ===================================================================
print("\n" + "=" * 65)
print("SECTION 2: Data Augmentation Pipeline")
print("=" * 65)

TRAIN_N, VAL_N = 15000, 3000
x_tr  = x_train[:TRAIN_N];    y_tr  = y_train[:TRAIN_N]
x_val = x_train[TRAIN_N:TRAIN_N + VAL_N];  y_val = y_train[TRAIN_N:TRAIN_N + VAL_N]

def augment(image, label):
    """Apply random augmentations during training for regularisation."""
    image = tf.image.random_flip_left_right(image)               # horizontal flip
    image = tf.image.random_brightness(image, max_delta=0.2)     # brightness jitter
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # contrast jitter
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # Random crop: pad by 4 pixels then crop back
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 8, IMG_SIZE + 8)
    image = tf.image.random_crop(image, [IMG_SIZE, IMG_SIZE, 3])
    return image, label

def preprocess(images, labels):
    images = tf.cast(images, tf.float32)
    images = tf.image.resize(images, [IMG_SIZE, IMG_SIZE])
    images = mobilenet_preprocess(images)
    labels = tf.squeeze(labels, -1)
    labels = tf.one_hot(labels, NUM_CLASSES)
    return images, labels

def preprocess_with_aug(images, labels):
    images, labels = preprocess(images, labels)
    images, labels = augment(images, labels)
    return images, labels

train_ds = (tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
            .map(preprocess_with_aug, num_parallel_calls=tf.data.AUTOTUNE)
            .shuffle(3000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

val_ds   = (tf.data.Dataset.from_tensor_slices((x_val, y_val))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

test_ds  = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
            .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE))

print(f"Train pipeline: {TRAIN_N} images with augmentation (flip, brightness, contrast, crop)")
print(f"Val/Test pipeline: no augmentation")

# Visualise augmented images
sample_imgs = x_tr[:8]
sample_lbls = y_tr[:8]
aug_results = []
for img, lbl in zip(sample_imgs, sample_lbls):
    img_f = tf.cast(img, tf.float32)
    img_r = tf.image.resize(img_f, [IMG_SIZE, IMG_SIZE])
    img_a, _ = augment(img_r, lbl)
    aug_results.append(img_a.numpy().clip(0, 255).astype(np.uint8))

fig, axes = plt.subplots(2, 8, figsize=(16, 5))
fig.suptitle('Original vs Augmented Samples', fontsize=13, fontweight='bold')
for i in range(8):
    orig = np.array(tf.image.resize(tf.cast(sample_imgs[i], tf.float32), [IMG_SIZE, IMG_SIZE]),
                    dtype=np.uint8)
    axes[0, i].imshow(orig); axes[0, i].axis('off')
    axes[0, i].set_title(CLASS_NAMES[sample_lbls[i][0]], fontsize=8)
    axes[1, i].imshow(aug_results[i]); axes[1, i].axis('off')
axes[0, 0].set_ylabel('Original', fontsize=10)
axes[1, 0].set_ylabel('Augmented', fontsize=10)
plt.tight_layout()
plt.savefig('lesson_20_augmentation.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_20_augmentation.png")

# ===================================================================
# SECTION 3: TRANSFER LEARNING + FINE-TUNING
# ===================================================================
print("\n" + "=" * 65)
print("SECTION 3: Transfer Learning + Fine-Tuning")
print("=" * 65)

# --- Stage 1: Feature extraction ---
base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
base.trainable = False

inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x   = base(inp, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.Dense(256, activation='relu')(x)
x   = layers.Dropout(0.5)(x)
out = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = keras.Model(inp, out, name='capstone_model')

model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nStage 1 — Feature Extraction")
print(f"Trainable params: {sum(np.prod(v.shape) for v in model.trainable_weights):,}")

t0 = time.time()
history_s1 = model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ], verbose=1
)
t1 = time.time()
_, acc_s1 = model.evaluate(test_ds, verbose=0)
print(f"Stage 1 test accuracy: {acc_s1*100:.2f}%  (time: {t1-t0:.0f}s)")

# --- Stage 2: Fine-tune top block ---
mn_base = None
for layer in model.layers:
    if 'mobilenetv2' in layer.name:
        mn_base = layer; break

if mn_base:
    mn_base.trainable = True
    unfreeze_from = len(mn_base.layers) - 20
    for i, layer in enumerate(mn_base.layers):
        if i < unfreeze_from:
            layer.trainable = False
        elif isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(f"\nStage 2 — Fine-Tuning (top block, LR=1e-4)")
print(f"Trainable params: {sum(np.prod(v.shape) for v in model.trainable_weights):,}")

history_s2 = model.fit(
    train_ds, validation_data=val_ds, epochs=15,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ], verbose=1
)
t2 = time.time()
_, acc_s2 = model.evaluate(test_ds, verbose=0)
print(f"Stage 2 test accuracy: {acc_s2*100:.2f}%  (time: {t2-t1:.0f}s)")

# ===================================================================
# SECTION 4: EVALUATION, GRAD-CAM, ERROR ANALYSIS, AND REPORT
# ===================================================================
print("\n" + "=" * 65)
print("SECTION 4: Full Evaluation, Grad-CAM, Error Analysis & Report")
print("=" * 65)

# --- Full evaluation ---
print("\nComputing predictions on full test set...")
y_pred_probs = model.predict(test_ds, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_te_flat

print("\nClassification Report:")
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=3)
print(report)

# Confusion matrix
cm_arr = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm_arr, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted', fontsize=12); ax.set_ylabel('True', fontsize=12)
ax.set_title('Confusion Matrix — Capstone Model', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('lesson_20_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_20_confusion_matrix.png")

# --- Learning curves ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Capstone Training History', fontsize=13, fontweight='bold')
all_val_acc  = history_s1.history['val_accuracy'] + history_s2.history['val_accuracy']
all_train_acc= history_s1.history['accuracy']     + history_s2.history['accuracy']
all_val_loss = history_s1.history['val_loss']     + history_s2.history['val_loss']
all_train_loss=history_s1.history['loss']         + history_s2.history['loss']
s1_len = len(history_s1.history['accuracy'])

axes[0].plot(all_train_acc, label='Train Acc', color='blue')
axes[0].plot(all_val_acc,   label='Val Acc',   color='blue', linestyle='--')
axes[0].axvline(s1_len - 1, color='red', linestyle=':', linewidth=2, label='Fine-tuning start')
axes[0].set_title('Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(all_train_loss, label='Train Loss', color='orange')
axes[1].plot(all_val_loss,   label='Val Loss',   color='orange', linestyle='--')
axes[1].axvline(s1_len - 1, color='red', linestyle=':', linewidth=2, label='Fine-tuning start')
axes[1].set_title('Loss'); axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('lesson_20_training_history.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_20_training_history.png")

# --- Grad-CAM on sample images ---
def gradcam_heatmap(model, img_array, last_conv_name, pred_index=None):
    """Grad-CAM using GradientTape (same as Lesson 18)."""
    mn_layer = None
    for lyr in model.layers:
        if 'mobilenetv2' in lyr.name:
            mn_layer = lyr; break
    if mn_layer is None:
        return np.zeros((7, 7)), 0, 0.0
    try:
        target = mn_layer.get_layer(last_conv_name)
        gmodel = keras.Model(model.inputs, [target.output, model.output])
    except Exception:
        return np.zeros((7, 7)), 0, 0.0
    with tf.GradientTape() as tape:
        conv_out, preds = gmodel(img_array, training=False)
        tape.watch(conv_out)
        if pred_index is None:
            pred_index = int(tf.argmax(preds[0]))
        score = preds[:, pred_index]
    grads = tape.gradient(score, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()
    fmaps  = conv_out[0].numpy()
    for k in range(pooled.shape[0]):
        fmaps[:, :, k] *= pooled[k]
    hm = np.maximum(np.mean(fmaps, axis=-1), 0)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    return hm, pred_index, float(preds[0][pred_index])

def overlay_heatmap(raw_uint8, hm, alpha=0.45):
    h, w = raw_uint8.shape[:2]
    hm_r = np.array(tf.image.resize(hm[..., np.newaxis], [h, w]))[:, :, 0]
    col  = (cm.get_cmap('jet')(hm_r)[:, :, :3] * 255).astype(np.uint8)
    return np.clip((1 - alpha) * raw_uint8 + alpha * col, 0, 255).astype(np.uint8)

# Find last conv name in MobileNetV2 base
last_conv_name = 'Conv_1'
if mn_base is not None:
    for sl in reversed(mn_base.layers):
        if isinstance(sl, layers.Conv2D):
            last_conv_name = sl.name; break

# Select correct and incorrect predictions for Grad-CAM
correct_idx   = np.where(y_pred == y_true)[0][:3]
incorrect_idx = np.where(y_pred != y_true)[0][:3]
all_gc_idx    = list(correct_idx) + list(incorrect_idx)
labels_gc     = ['✓ Correct'] * 3 + ['✗ Wrong'] * 3

print(f"\nGenerating Grad-CAM for {len(all_gc_idx)} images...")

def preprocess_single(raw_img):
    img = tf.cast(raw_img, tf.float32)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = mobilenet_preprocess(img)
    return tf.expand_dims(img, 0).numpy()

fig, axes = plt.subplots(len(all_gc_idx), 3, figsize=(10, 4 * len(all_gc_idx)))
fig.suptitle('Grad-CAM — Correct and Misclassified Examples', fontsize=13, fontweight='bold')
for row, (idx, tag) in enumerate(zip(all_gc_idx, labels_gc)):
    raw   = x_test[idx]
    inp   = preprocess_single(raw)
    raw_b = np.array(tf.image.resize(raw[np.newaxis], [IMG_SIZE, IMG_SIZE])[0], dtype=np.uint8)
    hm, pi, conf = gradcam_heatmap(model, inp, last_conv_name)
    ovl  = overlay_heatmap(raw_b, hm)
    axes[row, 0].imshow(raw_b);  axes[row, 0].axis('off')
    axes[row, 0].set_ylabel(f'{tag}\nTrue: {CLASS_NAMES[y_true[idx]]}', fontsize=8)
    axes[row, 1].imshow(hm, cmap='jet'); axes[row, 1].axis('off')
    axes[row, 1].set_title(f'Pred: {CLASS_NAMES[pi]} ({conf*100:.1f}%)', fontsize=8)
    axes[row, 2].imshow(ovl);   axes[row, 2].axis('off')
plt.tight_layout()
plt.savefig('lesson_20_gradcam.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson_20_gradcam.png")

# --- Error analysis ---
print("\nError Analysis:")
wrong_idx   = np.where(y_pred != y_true)[0]
error_pairs = {}
for wi in wrong_idx:
    pair = (CLASS_NAMES[y_true[wi]], CLASS_NAMES[y_pred[wi]])
    error_pairs[pair] = error_pairs.get(pair, 0) + 1
top_errors = sorted(error_pairs.items(), key=lambda kv: -kv[1])[:10]
print(f"  Total errors: {len(wrong_idx)} / {len(y_true)}  ({len(wrong_idx)/len(y_true)*100:.1f}%)")
print(f"\n  Top confusion pairs (true → predicted):")
for (true_c, pred_c), cnt in top_errors:
    print(f"    {true_c:<12} → {pred_c:<12}: {cnt:4d} errors")

# --- Save final model ---
model.save('lesson_20_capstone_model.keras')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_bytes = converter.convert()
with open('lesson_20_capstone.tflite', 'wb') as f:
    f.write(tflite_bytes)
keras_size  = os.path.getsize('lesson_20_capstone_model.keras') / 1e6
tflite_size = os.path.getsize('lesson_20_capstone.tflite') / 1e6
print(f"\nSaved Keras model: lesson_20_capstone_model.keras ({keras_size:.1f} MB)")
print(f"Saved TFLite model: lesson_20_capstone.tflite ({tflite_size:.1f} MB)")

# --- Comprehensive model report ---
from sklearn.metrics import f1_score, accuracy_score
macro_f1   = f1_score(y_true, y_pred, average='macro')
weighted_f1= f1_score(y_true, y_pred, average='weighted')
per_class_f1 = f1_score(y_true, y_pred, average=None)

print("\n" + "=" * 65)
print("CAPSTONE MODEL REPORT")
print("=" * 65)
print(f"Architecture:       MobileNetV2 (ImageNet) + custom head")
print(f"Training images:    {TRAIN_N:,} (with augmentation)")
print(f"Validation images:  {VAL_N:,}")
print(f"Test images:        {len(y_true):,}")
print(f"Input size:         {IMG_SIZE}×{IMG_SIZE}×3")
print(f"Preprocessing:      MobileNetV2 normalisation ([-1, 1])")
print(f"")
print(f"Stage 1 (Feature Extraction) accuracy: {acc_s1*100:.2f}%")
print(f"Stage 2 (Fine-Tuned) accuracy:         {acc_s2*100:.2f}%")
print(f"Final test accuracy:                   {accuracy_score(y_true, y_pred)*100:.2f}%")
print(f"Macro F1 score:                        {macro_f1:.4f}")
print(f"Weighted F1 score:                     {weighted_f1:.4f}")
print(f"")
print(f"Per-class F1 scores:")
for i, name in enumerate(CLASS_NAMES):
    bar = '█' * int(per_class_f1[i] * 20)
    print(f"  {name:<12}: {per_class_f1[i]:.3f}  {bar}")
print(f"")
print(f"Keras model size:   {keras_size:.1f} MB")
print(f"TFLite model size:  {tflite_size:.1f} MB ({tflite_size/keras_size*100:.0f}% of Keras size)")
print(f"")
print(f"Limitations:")
print(f"  - Tested only on CIFAR-10 distribution (32×32 upscaled)")
print(f"  - Performance on out-of-distribution images may degrade")
print(f"  - Highest confusion: see top error pairs above")
print(f"")
print(f"Saved outputs: lesson_20_eda.png, lesson_20_augmentation.png,")
print(f"               lesson_20_confusion_matrix.png, lesson_20_training_history.png,")
print(f"               lesson_20_gradcam.png, lesson_20_capstone_model.keras,")
print(f"               lesson_20_capstone.tflite")
print("=" * 65)
print("LESSON 20 AND COURSE COMPLETE — Congratulations!")
print("You have built a full end-to-end CNN image classification pipeline.")
print("=" * 65)
