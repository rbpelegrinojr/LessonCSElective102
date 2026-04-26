# Lesson 12: Evaluating Your Model — Metrics & Confusion Matrix

## Learning Objectives

By the end of this lesson, you will be able to:

- Explain why accuracy alone can be a misleading metric for model evaluation
- Construct and interpret a confusion matrix for binary and multi-class problems
- Calculate precision, recall, and F1 score from first principles
- Understand the precision-recall tradeoff and when to optimize for each
- Generate and interpret ROC curves and AUC scores
- Produce a full classification report and explain macro, micro, and weighted averaging

---

## Detailed Explanation

### Why Accuracy Alone Is Misleading

Accuracy is defined as: `(Number of correct predictions) / (Total predictions)`. It seems straightforward, but it fails spectacularly on **imbalanced datasets**. Consider a medical test for a rare disease that affects 1% of the population. A model that simply predicts "no disease" for every patient achieves **99% accuracy** — yet it is completely useless because it never identifies a single sick patient.

This is why professional data scientists almost never report accuracy alone. You need metrics that give you visibility into the types of errors the model is making.

### The Confusion Matrix

The **confusion matrix** is the foundation of all classification metrics. For a binary problem, it is a 2×2 table:

```
                  Predicted: Positive   Predicted: Negative
Actual: Positive       TP (True Pos)        FN (False Neg)
Actual: Negative       FP (False Pos)       TN (True Neg)
```

- **True Positive (TP):** Model predicted positive, and it actually was positive. ✅ Correct
- **True Negative (TN):** Model predicted negative, and it actually was negative. ✅ Correct
- **False Positive (FP):** Model predicted positive, but it was actually negative. ❌ Type I Error (false alarm)
- **False Negative (FN):** Model predicted negative, but it was actually positive. ❌ Type II Error (miss)

#### ASCII Confusion Matrix Example

```
              PREDICTED
              Cat    Dog
ACTUAL Cat  [  42     8 ]   ← 8 cats mistakenly called "dog"
       Dog  [   5    45 ]   ← 5 dogs mistakenly called "cat"
```

For multi-class problems, the matrix extends to N×N where N is the number of classes. Diagonal elements are correct predictions; off-diagonal elements reveal which classes are being confused with each other.

### Precision: Of All Positive Predictions, How Many Were Right?

**Precision** answers: "When the model says 'yes', how often is it correct?"

```
Precision = TP / (TP + FP)
```

Example: If a spam filter flags 100 emails as spam, and 90 of them are actually spam while 10 are legitimate — precision is 90/100 = 0.90. High precision means few false alarms.

### Recall (Sensitivity): Of All Actual Positives, How Many Did We Find?

**Recall** (also called **sensitivity** or **true positive rate**) answers: "Of all the actual positives, what fraction did the model correctly identify?"

```
Recall = TP / (TP + FN)
```

Example: If there are 200 actual spam emails and the model identifies 180 of them — recall is 180/200 = 0.90. High recall means few misses.

### F1 Score: Harmonic Mean of Precision and Recall

Neither precision nor recall alone is sufficient. The **F1 Score** combines both into a single number using the harmonic mean (which penalizes extreme imbalance between the two):

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean ensures that a model cannot game the F1 score by maximizing one metric while ignoring the other. If precision = 1.0 and recall = 0.0, F1 = 0.0 (correctly penalized).

The **Fβ score** is a generalization: `Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)`. Setting `β > 1` weights recall more heavily; `β < 1` weights precision more heavily.

### Precision-Recall Tradeoff

There is an inherent tension between precision and recall. By changing the **decision threshold** (the probability cutoff above which the model predicts "positive"), you can trade one for the other:

- Lowering the threshold → more items predicted positive → higher recall, lower precision
- Raising the threshold → fewer items predicted positive → lower recall, higher precision

The **Precision-Recall curve** plots precision vs recall at every possible threshold. The **average precision (AP)** summarizes this curve as a single number (area under the PR curve).

### ROC Curve and AUC

The **Receiver Operating Characteristic (ROC) curve** plots:
- **Y-axis:** True Positive Rate (TPR) = Recall = `TP / (TP + FN)`
- **X-axis:** False Positive Rate (FPR) = `FP / (FP + TN)`

at every possible decision threshold. A perfect classifier has a point at (0, 1) — zero false positives, 100% recall. A random classifier follows the diagonal line (slope = 1).

**AUC** (Area Under the ROC Curve) summarizes the curve into a single number between 0 and 1:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: No better than random
- AUC = 0.0: Perfectly wrong (just flip predictions)

For multi-class problems, you compute ROC curves using a **one-vs-rest** approach: treat each class as "positive" and all others as "negative".

### Top-K Accuracy

For problems with a large number of classes (like ImageNet with 1000 classes), **Top-K accuracy** is used. Instead of requiring the model to pick the single correct class, it is considered correct if the true label appears in the model's top K predictions. Top-5 accuracy is standard for ImageNet benchmarks.

```
Top-5 accuracy = Fraction of examples where true class is in model's 5 highest probability predictions
```

### Per-Class Metrics and Macro/Micro/Weighted Averaging

When dealing with multi-class problems, you compute precision, recall, and F1 for each class individually, then aggregate:

- **Macro averaging:** Compute the metric for each class, then take the unweighted mean. Every class contributes equally, regardless of how many examples it has. Use this when all classes are equally important.

- **Micro averaging:** Pool all TP, FP, FN across all classes, then compute the metric. Dominated by the performance on large classes. Equivalent to accuracy when there is no class imbalance.

- **Weighted averaging:** Same as macro but each class is weighted by its support (number of true instances). Accounts for class imbalance while still penalizing poor performance on minority classes.

### When to Optimize for Precision vs Recall

The choice depends on the **cost of different error types**:

- **Medical diagnosis (cancer screening):** A false negative (missing a real cancer) is catastrophic. Optimize for **high recall**, even at the cost of precision (more biopsies for benign cases).
- **Spam filtering:** A false positive (flagging a legitimate email as spam) is disruptive. Optimize for **high precision** — only flag email you're very confident is spam.
- **Fraud detection:** Missing fraud (FN) is expensive, but too many false alarms (FP) creates friction for customers. Requires balancing based on business cost.

### Classification Report Format

The `sklearn.metrics.classification_report` function produces a readable summary:

```
              precision    recall  f1-score   support

    airplane       0.73      0.78      0.75      1000
  automobile       0.85      0.84      0.85      1000
        bird       0.61      0.55      0.58      1000
         cat       0.52      0.48      0.50      1000

    accuracy                           0.73     10000
   macro avg       0.68      0.68      0.68     10000
weighted avg       0.73      0.73      0.73     10000
```

Support = number of true instances in that class. A large gap between precision and recall for a class indicates systematic bias. Classes with low support and low F1 are candidates for more training data.

---

## Key Concepts Table

| Metric | Formula | Best Use Case |
|---|---|---|
| Accuracy | (TP + TN) / Total | Balanced classes, quick overview |
| Precision | TP / (TP + FP) | When false positives are costly |
| Recall | TP / (TP + FN) | When false negatives are costly |
| F1 Score | 2 * P * R / (P + R) | Imbalanced classes |
| AUC-ROC | Area under ROC curve | General discriminative ability |
| Top-K Accuracy | True class in top K predictions | Large number of classes |
| Macro Avg | Mean of per-class metrics | Equal class importance |
| Weighted Avg | Support-weighted per-class metrics | Imbalanced but realistic |

---

## Code Reference

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_auc_score
)
import seaborn as sns

# Generate predictions
y_pred = model.predict(X_test).argmax(axis=1)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

# Classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Per-class precision, recall, F1
p, r, f1, support = precision_recall_fscore_support(y_true, y_pred)
```

---

## Activities

1. **Confusion Matrix:** Train a CIFAR-10 classifier for 10 epochs. Generate predictions on the test set and plot the confusion matrix using `sklearn.metrics.ConfusionMatrixDisplay`. Identify and print the two most frequently confused class pairs.

2. **Per-Class Metrics:** Using `sklearn.metrics.classification_report`, compute per-class precision, recall, and F1-score for your CIFAR-10 model. Print the report and write code to extract and display the class with the lowest F1-score.
## Review Questions

1. Why can a classifier with 99% accuracy be completely useless? Describe a concrete example.
2. In a medical diagnosis context, which error type (FP or FN) is typically more costly? How does this affect which metric to optimize?
3. Why is F1 score preferred over a simple average of precision and recall?
4. What does an AUC of 0.5 tell you about a classifier?
5. Explain the difference between macro and weighted averaging. In which scenario would you prefer each?
6. How does raising the decision threshold affect precision and recall simultaneously?
7. What is top-5 accuracy and why is it used for ImageNet-scale problems?

---

## Further Reading

- Sokolova, M., & Lapalme, G. (2009). *A systematic analysis of performance measures for classification tasks*. Information Processing & Management.
- Fawcett, T. (2006). *An introduction to ROC analysis*. Pattern Recognition Letters.
- scikit-learn metrics documentation: https://scikit-learn.org/stable/modules/model_evaluation.html
- Davis, J., & Goadrich, M. (2006). *The relationship between Precision-Recall and ROC curves*. ICML.
