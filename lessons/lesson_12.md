# Lesson 12: Evaluating Your Model — Metrics and Confusion Matrix

## Learning Objectives
- Explain why accuracy alone is an insufficient metric, especially under class imbalance
- Calculate precision, recall, and F1-score from first principles using TP, FP, FN, TN counts
- Interpret a confusion matrix and identify which classes the model confuses most
- Construct and interpret an ROC curve and explain what the AUC value represents
- Distinguish between macro, micro, and weighted averaging strategies for multi-class metrics
- Choose the right evaluation metric given a domain-specific requirement (e.g., medical diagnosis vs. spam filtering)

---

## Detailed Explanation

### Why Accuracy Is Not Enough

Suppose you build a model to detect a rare disease that affects 1% of the population. A naive model that simply predicts "healthy" for everyone achieves 99% accuracy — yet it is completely useless because it never identifies a single sick patient. This is the **class imbalance problem**, and it illustrates why accuracy alone can be dangerously misleading.

Accuracy is defined as:
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

Where:
- **TP** (True Positive): Correctly predicted positive
- **TN** (True Negative): Correctly predicted negative
- **FP** (False Positive): Incorrectly predicted positive (Type I error)
- **FN** (False Negative): Incorrectly predicted negative (Type II error)

When classes are imbalanced, a model can achieve high accuracy by mostly ignoring minority classes. Metrics like precision, recall, and F1-score give a more nuanced picture.

---

### Precision: Don't Cry Wolf

**Precision** answers: "Of all the times the model cried 'positive', how often was it right?"

```
Precision = TP / (TP + FP)
```

High precision means: when the model says something IS the target class, it is usually correct. Low precision means the model raises too many false alarms. In a spam filter, high precision is desirable — you don't want legitimate emails landing in the spam folder (false positives).

---

### Recall (Sensitivity): Don't Miss the Sick Ones

**Recall** answers: "Of all the actual positives, how many did the model find?"

```
Recall = TP / (TP + FN)
```

High recall means the model catches most true positives, even at the cost of some false alarms. In medical diagnosis (cancer screening), high recall is critical — it is far worse to miss a cancer case (false negative) than to trigger an unnecessary follow-up test (false positive).

**The Precision-Recall Tradeoff**: Improving precision often reduces recall and vice versa. By adjusting the decision threshold (e.g., from 0.5 to 0.7), you can shift this balance based on domain requirements.

---

### F1-Score: The Harmonic Mean

The **F1-score** combines precision and recall into a single number using the harmonic mean:

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

The harmonic mean punishes extreme imbalance between precision and recall more than the arithmetic mean does. If either precision or recall is very low, F1 is pulled down sharply. F1 = 1.0 is perfect; F1 = 0.0 is the worst.

The **general Fβ-score** allows you to weight recall β times more than precision:
```
Fβ = (1 + β²) * (Precision * Recall) / (β² * Precision + Recall)
```
Use β > 1 when recall matters more (medical); β < 1 when precision matters more (spam).

---

### The Confusion Matrix

A **confusion matrix** is a square grid that shows exactly what the model predicted vs. what the true labels were for every class. For a 3-class problem:

```
                   Predicted
                Cat  Dog  Bird
Actual  Cat  [  45    3    2  ]
        Dog  [   5   40    5  ]
        Bird [   1    4   45  ]
```

Reading this matrix:
- Row = true class, Column = predicted class
- Diagonal cells = correct predictions
- Off-diagonal cells = errors

For class "Dog": TP=40, FP=3+4=7, FN=5+5=10, TN=(45+2+1+4+45)=97. This lets you compute per-class precision, recall, and F1.

The confusion matrix is the richest evaluation artifact available. It immediately reveals systematic confusions — perhaps the model always confuses dogs and cats but is excellent at birds. This guides where to improve your model or gather more training data.

---

### ROC Curve and AUC

The **ROC (Receiver Operating Characteristic) curve** plots:
- **TPR (True Positive Rate = Recall)** on the Y-axis
- **FPR (False Positive Rate = FP/(FP+TN))** on the X-axis

As you vary the classification threshold from 0 to 1, the model's TPR and FPR trace out a curve. A perfect classifier reaches the top-left corner (TPR=1, FPR=0). A random classifier produces a diagonal line from (0,0) to (1,1).

**AUC (Area Under the Curve)** summarizes the entire ROC curve in one number:
- AUC = 1.0: Perfect classifier
- AUC = 0.5: Random guessing
- AUC = 0.0: Perfectly wrong (flip predictions!)

AUC is threshold-independent, making it useful when you haven't committed to a specific decision threshold. For multi-class problems, AUC is computed per-class in a one-vs-rest manner.

```
ROC Curve Diagram:
TPR
1.0|        *----*
   |      *      \
   |    *          \
   |  *              * 
0.5|*       Random    \
   |                    *
0.0+---*--*--*----------*
   0.0                 1.0
                      FPR
```

---

### Top-K Accuracy

For problems with many classes (ImageNet has 1,000), **Top-5 accuracy** is often reported: the model scores a "hit" if the true class appears in its 5 highest-probability predictions. This is more forgiving than Top-1 accuracy and better reflects cases where multiple answers are visually plausible.

---

### Per-Class Metrics and Averaging Strategies

When evaluating multi-class classifiers:

**Macro Averaging**: Compute the metric for each class independently, then take the unweighted mean. Treats all classes equally, regardless of how many samples they have. Best when all classes are equally important.

```
Macro F1 = (F1_class1 + F1_class2 + ... + F1_classN) / N
```

**Micro Averaging**: Aggregate TP, FP, FN across all classes first, then compute the metric. Gives more weight to frequent classes. For accuracy-like behavior.

**Weighted Averaging**: Like macro but weights each class by its support (number of samples). Accounts for class imbalance while reporting a single number.

```
Weighted F1 = Σ (F1_class_i * support_i) / total_samples
```

---

### When to Optimize Precision vs. Recall

| Domain | Prefer | Reason |
|--------|--------|--------|
| Cancer screening | High Recall | Missing a positive diagnosis is catastrophic |
| Spam filtering | High Precision | False positives (lost emails) are costly |
| Fraud detection | High Recall | Missing fraud is more damaging than false alerts |
| News recommendation | High Precision | Irrelevant articles annoy users |
| Drug side-effect detection | High Recall | Missing side effects can harm patients |

Understanding the asymmetric cost of errors is the most important skill in applied machine learning evaluation. Always ask: "What's worse — a false positive or a false negative?"

---

### Common Misconceptions

1. **"High accuracy means good model"** — Only if classes are balanced. Always check per-class metrics and the confusion matrix.
2. **"AUC is always the best metric"** — AUC is threshold-independent, which is great for exploratory analysis but doesn't reflect actual deployed performance at a specific threshold.
3. **"Micro and macro F1 should be close"** — Only if classes are balanced. A big gap signals class imbalance issues.
4. **"Precision and recall are both maximized together"** — There is an inherent tradeoff. You can only maximize one at a time unless your model is perfect.

---

## Key Concepts

| Term | Definition | Why It Matters |
|------|-----------|----------------|
| Accuracy | (TP+TN) / total predictions | Intuitive but misleading under imbalance |
| True Positive (TP) | Model correctly predicts positive class | Building block for all other metrics |
| False Positive (FP) | Model incorrectly predicts positive | Drives down precision; costly in spam filters |
| False Negative (FN) | Model misses a true positive | Drives down recall; dangerous in medical diagnosis |
| Precision | TP / (TP + FP) | How trustworthy positive predictions are |
| Recall | TP / (TP + FN) | How complete positive predictions are |
| F1-Score | Harmonic mean of precision and recall | Single balanced metric for imbalanced datasets |
| Confusion Matrix | Grid of predicted vs. actual classes | Reveals per-class errors and systematic confusions |
| ROC Curve | TPR vs. FPR at different thresholds | Threshold-independent model comparison |
| AUC | Area under the ROC curve | Summary metric: 1.0=perfect, 0.5=random |
| Top-K Accuracy | True label in top-K predictions | Useful for many-class problems like ImageNet |
| Macro Average | Unweighted mean of per-class metrics | Treats all classes equally |
| Weighted Average | Support-weighted mean of per-class metrics | Accounts for class frequency |

---

## Code Reference

See the full runnable demo in **code/lesson_12.py**

---

## Activities

1. **Imbalance Experiment**: Create a synthetic binary dataset with 95% negative and 5% positive. Train a logistic regression model and report accuracy, precision, recall, and F1. What does accuracy hide that F1 reveals?

2. **Confusion Matrix Analysis**: Train a CNN on CIFAR-10 for 10 epochs. Plot the confusion matrix as a heatmap. Identify the top-3 most confused class pairs. Explain why the model might confuse those specific classes.

3. **Threshold Manipulation**: Using a binary classifier, change the decision threshold from 0.3 to 0.5 to 0.7. Compute and compare precision, recall, and F1 at each threshold. Which threshold would you choose for a medical screening task vs. a spam filter?

4. **ROC Curve Comparison**: Train two different model architectures (e.g., a shallow CNN and a deeper CNN) on CIFAR-10. Plot both ROC curves for a single class (e.g., "automobile") on the same figure. Compute and compare their AUC values.

5. **Macro vs. Weighted F1**: On an imbalanced version of MNIST (subsample some digits to be rare), compute both macro and weighted F1. Explain the gap between the two numbers in terms of which classes are being underserved.

---

## Review Questions

1. A model achieves 98% accuracy on a medical dataset. Should the development team be confident? What additional information do you need?
2. Explain the precision-recall tradeoff using a concrete example. How would you choose the optimal threshold for a cancer detection system?
3. How do you read a confusion matrix? Given a 4x4 confusion matrix, how would you compute precision and recall for class 2?
4. What does an AUC of 0.5 tell you about a binary classifier? What does AUC of 0.95 suggest about generalization?
5. When would you prefer macro F1 over weighted F1? Give a concrete domain example.

---

## Further Reading

- Scikit-learn documentation: `sklearn.metrics` — precision, recall, F1, confusion matrix, ROC AUC
- "Beyond Accuracy: Precision and Recall" — Towards Data Science article series
- "The Relationship Between Precision-Recall and ROC Curves" — Jesse Davis & Mark Goadrich, 2006
- "Imbalanced-Learn: A Python Toolbox to Tackle the Curse of Imbalanced Datasets" — Lemaître et al., JMLR 2017
- Deep Learning Specialization (Coursera), Course 3: Structuring Machine Learning Projects
