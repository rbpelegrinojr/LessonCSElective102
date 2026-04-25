"""
Lesson 12: Evaluating Your Model — Metrics and Confusion Matrix
================================================================
This module demonstrates comprehensive model evaluation beyond accuracy.
It covers:
  - Why accuracy fails under class imbalance
  - Precision, recall, F1-score computation
  - Confusion matrix visualization with seaborn
  - ROC curves and AUC for multi-class classification
  - Per-class metrics and macro/micro/weighted averaging
  - Training a CNN on CIFAR-10 and evaluating with sklearn
"""

# === Standard library and framework imports ===
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10

# Scikit-learn evaluation metrics
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.preprocessing import label_binarize

# Seaborn for heatmap visualization
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("  Note: seaborn not installed; confusion matrix will use matplotlib directly")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("=" * 60)
print("Lesson 12: Evaluating Your Model — Metrics and Confusion Matrix")
print("=" * 60)
print(f"TensorFlow version: {tf.__version__}")

# CIFAR-10 class names for human-readable labels
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']


# ===========================================================
# === SECTION 1: WHY ACCURACY FAILS — CLASS IMBALANCE DEMO ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 1: The Class Imbalance Problem")
print("=" * 60)

# Create a synthetic imbalanced binary dataset
np.random.seed(42)
n_total = 1000
n_positive = 50      # Only 5% positive (rare disease, fraud, etc.)
n_negative = n_total - n_positive

# Simulate a naive "always predict negative" classifier
y_true_binary = np.array([1] * n_positive + [0] * n_negative)
y_pred_naive  = np.zeros(n_total, dtype=int)   # Always predicts 0 (negative)

# Calculate metrics for the naive classifier
accuracy_naive = np.mean(y_pred_naive == y_true_binary)
tp = np.sum((y_pred_naive == 1) & (y_true_binary == 1))
fp = np.sum((y_pred_naive == 1) & (y_true_binary == 0))
fn = np.sum((y_pred_naive == 0) & (y_true_binary == 1))
tn = np.sum((y_pred_naive == 0) & (y_true_binary == 0))

precision_naive = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall_naive    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1_naive        = (2 * precision_naive * recall_naive / (precision_naive + recall_naive)
                   if (precision_naive + recall_naive) > 0 else 0.0)

print(f"Imbalanced dataset: {n_positive} positive ({100*n_positive/n_total:.0f}%), "
      f"{n_negative} negative ({100*n_negative/n_total:.0f}%)")
print(f"\nNaive 'always predict negative' classifier:")
print(f"  Accuracy  = {accuracy_naive:.2%}  ← Looks great!")
print(f"  Precision = {precision_naive:.2%}  ← Undefined / zero")
print(f"  Recall    = {recall_naive:.2%}  ← Catches ZERO positives!")
print(f"  F1-Score  = {f1_naive:.2%}  ← Reveals the real problem")

# Visualize the confusion matrix for this naive example
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lesson 12: Why Accuracy Is Not Enough", fontsize=14, fontweight='bold')

# Confusion matrix
cm_naive = np.array([[tn, fp], [fn, tp]])
if HAS_SEABORN:
    sns.heatmap(cm_naive, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['Pred Negative', 'Pred Positive'],
                yticklabels=['True Negative', 'True Positive'])
else:
    im = axes[0].imshow(cm_naive, cmap='Blues')
    axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(['Pred Neg', 'Pred Pos'])
    axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(['True Neg', 'True Pos'])
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, str(cm_naive[i, j]), ha='center', va='center', fontsize=14)
    plt.colorbar(im, ax=axes[0])
axes[0].set_title(f"Naive Classifier CM\nAccuracy={accuracy_naive:.0%}, F1={f1_naive:.0%}")

# Bar chart of metrics
metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metric_vals  = [accuracy_naive, precision_naive, recall_naive, f1_naive]
bar_colors   = ['green' if v > 0.5 else 'red' for v in metric_vals]
bars = axes[1].bar(metric_names, metric_vals, color=bar_colors, edgecolor='black', alpha=0.8)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel("Score")
axes[1].set_title("Metrics for Naive Imbalanced Classifier\n(Green=misleadingly good, Red=reveals failure)")
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for bar, val in zip(bars, metric_vals):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.1%}", ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig("lesson_12_imbalance_demo.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_12_imbalance_demo.png")


# ===========================================================
# === SECTION 2: TRAIN CNN ON CIFAR-10 ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 2: Training CNN on CIFAR-10")
print("=" * 60)

# Load CIFAR-10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()   # Shape from (N,1) to (N,)
y_test  = y_test.flatten()

# Normalize pixel values to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32") / 255.0

print(f"Training set: {x_train.shape}, Test set: {x_test.shape}")

def build_cifar_cnn():
    """Build a small CNN for CIFAR-10 classification."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax'),
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

print("Building and training CIFAR-10 CNN (this may take a few minutes)...")
model = build_cifar_cnn()
model.summary()

history = model.fit(
    x_train, y_train,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")


# ===========================================================
# === SECTION 3: CONFUSION MATRIX AND PER-CLASS METRICS ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 3: Confusion Matrix and Per-Class Metrics")
print("=" * 60)

# Get model predictions on the test set
y_pred_probs = model.predict(x_test, verbose=0)    # Probability distributions
y_pred       = np.argmax(y_pred_probs, axis=1)      # Predicted class indices

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Normalize confusion matrix by row (true class) for percentage view
cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Lesson 12: CIFAR-10 Confusion Matrix", fontsize=14, fontweight='bold')

# Raw counts heatmap
if HAS_SEABORN:
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axes[0],
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES)
else:
    im0 = axes[0].imshow(cm, cmap='YlOrRd')
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_xticks(range(10)); axes[0].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    axes[0].set_yticks(range(10)); axes[0].set_yticklabels(CIFAR10_CLASSES)
    for i in range(10):
        for j in range(10):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=7)
axes[0].set_title("Confusion Matrix (Counts)")
axes[0].set_ylabel("True Class")
axes[0].set_xlabel("Predicted Class")
plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right")

# Normalized (percentage) heatmap
if HAS_SEABORN:
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                xticklabels=CIFAR10_CLASSES,
                yticklabels=CIFAR10_CLASSES,
                vmin=0, vmax=1)
else:
    im1 = axes[1].imshow(cm_normalized, cmap='Blues', vmin=0, vmax=1)
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_xticks(range(10)); axes[1].set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
    axes[1].set_yticks(range(10)); axes[1].set_yticklabels(CIFAR10_CLASSES)
axes[1].set_title("Confusion Matrix (Row-Normalized, Recall per Class)")
axes[1].set_ylabel("True Class")
axes[1].set_xlabel("Predicted Class")
plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("lesson_12_confusion_matrix.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_12_confusion_matrix.png")

# Print full classification report with per-class metrics
print("\nClassification Report (per-class precision, recall, F1):")
print(classification_report(y_test, y_pred, target_names=CIFAR10_CLASSES))

# Compute and display per-class metrics for bar chart
per_class_precision = precision_score(y_test, y_pred, average=None)
per_class_recall    = recall_score(y_test, y_pred, average=None)
per_class_f1        = f1_score(y_test, y_pred, average=None)

fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(CIFAR10_CLASSES))
width = 0.25

bars1 = ax.bar(x_pos - width, per_class_precision, width, label='Precision', color='steelblue',   alpha=0.85, edgecolor='black')
bars2 = ax.bar(x_pos,         per_class_recall,    width, label='Recall',    color='darkorange',  alpha=0.85, edgecolor='black')
bars3 = ax.bar(x_pos + width, per_class_f1,        width, label='F1-Score',  color='forestgreen', alpha=0.85, edgecolor='black')

ax.set_xlabel("CIFAR-10 Class")
ax.set_ylabel("Score")
ax.set_title("Lesson 12: Per-Class Precision, Recall, and F1-Score (CIFAR-10)")
ax.set_xticks(x_pos)
ax.set_xticklabels(CIFAR10_CLASSES, rotation=45, ha='right')
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=np.mean(per_class_f1), color='red', linestyle='--', linewidth=1.5,
           label=f'Mean F1 = {np.mean(per_class_f1):.3f}')
ax.legend()

plt.tight_layout()
plt.savefig("lesson_12_per_class_metrics.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_12_per_class_metrics.png")

# Report macro, micro, weighted averages
macro_f1    = f1_score(y_test, y_pred, average='macro')
micro_f1    = f1_score(y_test, y_pred, average='micro')
weighted_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"\nAveraging Strategies:")
print(f"  Macro F1    = {macro_f1:.4f}  (treats all classes equally)")
print(f"  Micro F1    = {micro_f1:.4f}  (equivalent to accuracy for balanced sets)")
print(f"  Weighted F1 = {weighted_f1:.4f}  (weighted by class support)")


# ===========================================================
# === SECTION 4: ROC CURVES AND AUC ===
# ===========================================================

print("\n" + "=" * 60)
print("SECTION 4: ROC Curves and AUC (One-vs-Rest)")
print("=" * 60)

# Binarize the test labels for one-vs-rest ROC computation
# This creates a 10-column binary matrix for each class
y_test_bin = label_binarize(y_test, classes=list(range(10)))

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    print(f"  AUC for class '{CIFAR10_CLASSES[i]}': {roc_auc[i]:.4f}")

# Plot all 10 ROC curves
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Lesson 12: ROC Curves (One-vs-Rest) for CIFAR-10", fontsize=14, fontweight='bold')

# Left: All 10 ROC curves on one plot
cmap = plt.cm.get_cmap('tab10', 10)
for i in range(10):
    axes[0].plot(fpr[i], tpr[i], color=cmap(i), linewidth=2,
                 label=f"{CIFAR10_CLASSES[i]} (AUC={roc_auc[i]:.2f})")
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC=0.50)')
axes[0].set_xlabel("False Positive Rate (FPR)")
axes[0].set_ylabel("True Positive Rate (TPR = Recall)")
axes[0].set_title("ROC Curves — All 10 Classes")
axes[0].legend(loc='lower right', fontsize=8)
axes[0].grid(True, alpha=0.3)

# Right: AUC bar chart per class
sorted_indices = np.argsort([roc_auc[i] for i in range(10)])
sorted_classes = [CIFAR10_CLASSES[i] for i in sorted_indices]
sorted_aucs    = [roc_auc[i] for i in sorted_indices]
bar_colors     = [cmap(i) for i in sorted_indices]

axes[1].barh(sorted_classes, sorted_aucs, color=bar_colors, edgecolor='black', alpha=0.85)
axes[1].axvline(x=0.5, color='red', linestyle='--', linewidth=1.5, label='Random AUC=0.5')
axes[1].axvline(x=1.0, color='green', linestyle='--', linewidth=1.5, label='Perfect AUC=1.0')
axes[1].set_xlabel("AUC Score")
axes[1].set_title("AUC per Class (One-vs-Rest)")
axes[1].set_xlim(0.4, 1.05)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='x')
for j, (cls, auc_val) in enumerate(zip(sorted_classes, sorted_aucs)):
    axes[1].text(auc_val + 0.002, j, f"{auc_val:.3f}", va='center', fontsize=9)

plt.tight_layout()
plt.savefig("lesson_12_roc_curves.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_12_roc_curves.png")
print(f"\nMean AUC across all classes: {np.mean(list(roc_auc.values())):.4f}")

# Training history plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Lesson 12: CIFAR-10 Training History", fontsize=14, fontweight='bold')

axes[0].plot(history.history['accuracy'],     'b-', label='Train Accuracy', linewidth=2)
axes[0].plot(history.history['val_accuracy'], 'r--', label='Val Accuracy',  linewidth=2)
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy")
axes[0].set_title("Training & Validation Accuracy"); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     'b-', label='Train Loss', linewidth=2)
axes[1].plot(history.history['val_loss'], 'r--', label='Val Loss',  linewidth=2)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title("Training & Validation Loss"); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lesson_12_training_history.png", dpi=100, bbox_inches='tight')
plt.show()
print("  → Saved: lesson_12_training_history.png")

print("\n" + "=" * 60)
print("Lesson 12 Complete!")
print("Generated files:")
print("  - lesson_12_imbalance_demo.png")
print("  - lesson_12_confusion_matrix.png")
print("  - lesson_12_per_class_metrics.png")
print("  - lesson_12_roc_curves.png")
print("  - lesson_12_training_history.png")
print("=" * 60)
