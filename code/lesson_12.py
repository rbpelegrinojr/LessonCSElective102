"""
Lesson 12: Evaluating Your Model — Metrics & Confusion Matrix
=============================================================
Demonstrates confusion matrix, precision/recall/F1, ROC curves,
and sklearn's classification_report on a CNN trained on CIFAR-10.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize

print("=" * 60)
print("LESSON 12: Evaluating Your Model — Metrics & Confusion Matrix")
print("=" * 60)

# Class names for CIFAR-10
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
N_CLASSES = len(CLASS_NAMES)

# ──────────────────────────────────────────────────────────────
# SECTION 1: Build and Evaluate CNN on CIFAR-10
# ──────────────────────────────────────────────────────────────
print("\n--- Section 1: Build and Evaluate CNN on CIFAR-10 ---")

# Load and normalise CIFAR-10
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32') / 255.0
y_train = y_train.flatten()   # shape (50000,)
y_test  = y_test.flatten()    # shape (10000,)

# Use a subset for training speed in demo
N_TRAIN = 15000
X_tr, y_tr = X_train[:N_TRAIN], y_train[:N_TRAIN]
X_val, y_val = X_train[N_TRAIN:N_TRAIN+3000], y_train[N_TRAIN:N_TRAIN+3000]

def build_cnn():
    """Small CNN for CIFAR-10 classification."""
    return keras.Sequential([
        keras.layers.Conv2D(32, 3, padding='same', activation='relu',
                            input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(N_CLASSES, activation='softmax'),
    ])

model = build_cnn()
model.compile(
    optimizer=keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()

print("\nTraining CNN (15 epochs, subset of CIFAR-10)...")
hist = model.fit(X_tr, y_tr,
                 validation_data=(X_val, y_val),
                 epochs=15, batch_size=64,
                 verbose=1)

# Obtain predictions on the test set (full 10 000 examples)
y_pred_prob = model.predict(X_test, verbose=0)          # shape (10000, 10)
y_pred      = y_pred_prob.argmax(axis=1)                # integer class predictions
test_acc    = np.mean(y_pred == y_test)
print(f"\nTest Accuracy: {test_acc:.4f}")

# ──────────────────────────────────────────────────────────────
# SECTION 2: Confusion Matrix with Seaborn Heatmap
# ──────────────────────────────────────────────────────────────
print("\n--- Section 2: Confusion Matrix ---")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix (rows=true, columns=predicted):")
print(cm)

# ASCII confusion matrix
print("\nASCII Confusion Matrix (top-left corner, first 5 classes):")
header = "       " + "  ".join(f"{c[:5]:>5}" for c in CLASS_NAMES[:5])
print(header)
for i in range(5):
    row_label = f"{CLASS_NAMES[i][:5]:>5}"
    row_vals  = "  ".join(f"{cm[i, j]:>5}" for j in range(5))
    print(f"{row_label}  {row_vals}")

# Plot full confusion matrix as heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('CIFAR-10 Confusion Matrix', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('lesson12_confusion_matrix.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson12_confusion_matrix.png")

# Find the most-confused class pair
np.fill_diagonal(cm, 0)          # zero out diagonal for off-diagonal analysis
most_confused = np.unravel_index(cm.argmax(), cm.shape)
print(f"\nMost confused pair: '{CLASS_NAMES[most_confused[0]]}' "
      f"predicted as '{CLASS_NAMES[most_confused[1]]}' "
      f"({cm[most_confused]} times)")

# ──────────────────────────────────────────────────────────────
# SECTION 3: Precision, Recall, F1 Per Class — Bar Chart
# ──────────────────────────────────────────────────────────────
print("\n--- Section 3: Per-Class Precision, Recall, F1 ---")

precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=range(N_CLASSES)
)

print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>6} {'Support':>9}")
print("-" * 50)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<12} {precision[i]:>10.4f} {recall[i]:>8.4f} {f1[i]:>6.4f} {support[i]:>9}")

# Compute aggregated metrics
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro'
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted'
)
print(f"\n{'Macro avg':<12} {prec_macro:>10.4f} {rec_macro:>8.4f} {f1_macro:>6.4f}")
print(f"{'Weighted avg':<12} {prec_weighted:>10.4f} {rec_weighted:>8.4f} {f1_weighted:>6.4f}")

# Plot per-class metrics as grouped bar chart
x = np.arange(N_CLASSES)
width = 0.28

fig, ax = plt.subplots(figsize=(13, 5))
bars1 = ax.bar(x - width, precision, width, label='Precision', color='steelblue')
bars2 = ax.bar(x,          recall,   width, label='Recall',    color='darkorange')
bars3 = ax.bar(x + width,  f1,       width, label='F1 Score',  color='forestgreen')

ax.set_xticks(x)
ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
ax.set_ylim(0, 1.05)
ax.set_ylabel('Score')
ax.set_title('Per-Class Precision, Recall, and F1 Score (CIFAR-10)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('lesson12_per_class_metrics.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson12_per_class_metrics.png")

# ──────────────────────────────────────────────────────────────
# SECTION 4: ROC Curves for Multi-Class (One-vs-Rest)
# ──────────────────────────────────────────────────────────────
print("\n--- Section 4: ROC Curves (One-vs-Rest) ---")

# Binarise the true labels (one-vs-rest)
y_test_bin = label_binarize(y_test, classes=range(N_CLASSES))  # (10000, 10)

fig, ax = plt.subplots(figsize=(9, 7))
colors = plt.cm.tab10(np.linspace(0, 1, N_CLASSES))

print("\n  Class              AUC")
print("  " + "-" * 30)

for i, (class_name, color) in enumerate(zip(CLASS_NAMES, colors)):
    fpr, tpr, thresholds = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=1.5,
            label=f'{class_name} (AUC = {roc_auc:.3f})')
    print(f"  {class_name:<18} {roc_auc:.4f}")

# Random classifier diagonal
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random (AUC = 0.5)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (FPR)')
ax.set_ylabel('True Positive Rate (TPR / Recall)')
ax.set_title('ROC Curves — CIFAR-10 (One-vs-Rest)')
ax.legend(loc='lower right', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lesson12_roc_curves.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved: lesson12_roc_curves.png")

# ──────────────────────────────────────────────────────────────
# SECTION 5: sklearn Classification Report
# ──────────────────────────────────────────────────────────────
print("\n--- Section 5: Full Classification Report ---")

report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
print("\nsklearn classification_report:")
print(report)

# --- Interpretation guide ---
print("\nREPORT INTERPRETATION GUIDE")
print("-" * 60)
print("precision : Of all predicted as this class, fraction correct")
print("recall    : Of all true instances, fraction correctly found")
print("f1-score  : Harmonic mean of precision and recall")
print("support   : Number of true instances in this class")
print()
print("accuracy  : Overall fraction of correct predictions")
print("macro avg : Unweighted mean across all classes")
print("            (treats all classes equally — use for class balance)")
print("weighted  : Support-weighted mean — accounts for class frequency")
print()

# Demonstrate why accuracy can be misleading with imbalanced data
print("IMBALANCED DATASET DEMO")
print("-" * 60)
# Simulate: 95% class 0, 5% class 1
np.random.seed(42)
n_samples = 1000
y_imb_true = np.concatenate([np.zeros(950), np.ones(50)]).astype(int)
# Naive classifier: always predict class 0
y_imb_pred_naive = np.zeros(n_samples, dtype=int)

naive_accuracy = np.mean(y_imb_true == y_imb_pred_naive)
_, _, naive_f1, _ = precision_recall_fscore_support(
    y_imb_true, y_imb_pred_naive, average='macro', zero_division=0
)
print(f"Naive 'always class 0' classifier:")
print(f"  Accuracy = {naive_accuracy:.4f}  ← looks great!")
print(f"  Macro F1 = {naive_f1:.4f}        ← reveals the truth")
print(f"  (F1 is 0 for class 1 because zero recall on minority class)")

print("\n" + "=" * 60)
print("Lesson 12 complete. Outputs saved:")
print("  lesson12_confusion_matrix.png")
print("  lesson12_per_class_metrics.png")
print("  lesson12_roc_curves.png")
print("=" * 60)
