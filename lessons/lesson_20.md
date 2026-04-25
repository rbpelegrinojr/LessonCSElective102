# Lesson 20: Capstone Project — End-to-End Image Classifier

## Learning Objectives
- Design and execute a complete end-to-end image classification pipeline from raw data through deployment-ready model
- Apply all course concepts in sequence: EDA, augmentation, transfer learning, fine-tuning, evaluation, and visualisation
- Perform systematic error analysis by examining misclassified examples to diagnose model weaknesses
- Write a professional model card documenting the model's capabilities, limitations, and deployment requirements
- Articulate a clear path forward in computer vision including object detection, segmentation, and vision transformers
- Identify and access high-quality resources for continuing your deep learning education independently

## Detailed Explanation

### Bringing It All Together

Over the past nineteen lessons you have accumulated a powerful toolkit:

```
Lesson 01–03:  Image data, pixels, colour spaces, NumPy
Lesson 04–06:  Building CNNs, convolution, pooling, activation functions
Lesson 07–09:  Training dynamics, loss functions, optimisers, learning rates
Lesson 10–12:  Regularisation, dropout, batch norm, data augmentation
Lesson 13–15:  Evaluation metrics, confusion matrices, class imbalance
Lesson 16–17:  Transfer learning, feature extraction, fine-tuning
Lesson 18:     Interpretability, Grad-CAM, visualisation
Lesson 19:     Saving, optimising, deploying models
Lesson 20:     ← YOU ARE HERE: Capstone — everything at once
```

This final lesson is about synthesis: taking all those individual skills and integrating them into a single, professional-quality project that you could show to a potential employer, include in a portfolio, or use as a foundation for a real-world application.

### Phase 1: Project Planning

Before writing a single line of code, professional ML engineers ask:

```
1. PROBLEM FRAMING
   - What is the prediction target? (classes, continuous value, multi-label?)
   - What business metric matters? (accuracy, recall, F1, latency?)
   - What is the cost of each error type? (false positive vs false negative)

2. DATA REQUIREMENTS
   - How many labelled examples per class are available?
   - Is the class distribution balanced?
   - Are there data quality issues? (duplicates, mislabels, corrupted files)
   - What augmentations are appropriate for this domain?

3. MODEL CONSTRAINTS
   - Where will the model run? (server GPU, mobile, browser?)
   - What is the latency budget? (50 ms? 200 ms? 1 s?)
   - What is the memory budget?

4. SUCCESS CRITERIA
   - What is the minimum acceptable accuracy?
   - How will you measure success in production?
```

### Phase 2: Exploratory Data Analysis (EDA)

EDA on image datasets involves:

**Distribution Analysis:**
```
Class Distribution:          Image Resolution Histogram:
airplane  ████ 1000          224×224: ████████████  8500
car       ████ 1000          192×192: ████           320
bird      ████ 1000          Other:   ██             180
cat       ████ 1000
...
```

**Visual Inspection:** Always look at samples from each class. Humans catch mislabels, near-duplicates, and domain surprises in minutes — things no statistical test would catch.

**Brightness and Contrast Statistics:** Compute mean pixel value and standard deviation per channel. Skewed brightness distribution suggests a need for normalisation or contrast augmentation.

**Difficulty Estimation:** Look at images your baseline gets wrong — these are usually the hardest, most ambiguous examples. Understanding them informs augmentation strategy.

### Phase 3: The Full Pipeline

```
Raw Images
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│                    PREPROCESSING PIPELINE                    │
│  Resize → Normalise → Augment (train only) → Batch          │
└──────────────────────┬───────────────────────────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │         TRANSFER LEARNING           │
    │  MobileNetV2 base (ImageNet)        │
    │  + Custom head                      │
    │  Stage 1: Feature extraction        │
    │  Stage 2: Fine-tune top blocks      │
    └──────────────────┬──────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │           EVALUATION                │
    │  Accuracy, F1, Confusion Matrix     │
    │  Grad-CAM visualisation             │
    │  Error analysis (misclassified)     │
    └──────────────────┬──────────────────┘
                       │
    ┌──────────────────▼──────────────────┐
    │         SAVE & OPTIMISE             │
    │  SavedModel + TFLite                │
    │  Model size & speed report          │
    └─────────────────────────────────────┘
```

### Phase 4: Hyperparameter Tuning Strategy

Systematic hyperparameter search, not random guessing:

| Hyperparameter | Range to Explore | Starting Point |
|---|---|---|
| Learning rate (head) | 1e-4 to 1e-2 | 1e-3 |
| Learning rate (fine-tune) | 1e-6 to 1e-4 | 1e-5 |
| Batch size | 16, 32, 64, 128 | 32 |
| Dropout rate | 0.2 to 0.6 | 0.5 |
| Dense head units | 128, 256, 512 | 256 |
| Augmentation strength | mild, moderate, heavy | moderate |

Use **Keras Tuner** or simple grid search for the most impactful parameters (learning rate, dropout). Never tune more than 2–3 hyperparameters at once unless you have a dedicated compute budget.

### Phase 5: Error Analysis

Error analysis is one of the highest-leverage activities in the ML development cycle. After getting initial results:

**Step 1: Collect Misclassified Examples**
```python
misclassified = [(img, true_label, pred_label)
                 for img, true_label, pred_label in test_set
                 if true_label != pred_label]
```

**Step 2: Categorise Errors**
- Same superclass confusion (cat ↔ dog, airplane ↔ bird)
- Lighting/angle/occlusion failures
- Rare or unusual examples
- Potential mislabels in the ground truth

**Step 3: Quantify Error Categories**
```
Error type                  Count    % of all errors
─────────────────────────────────────────────────────
Cat ↔ Dog confusion            42        28%
Dark/night images failing      31        21%
Partial occlusion              27        18%
Potential mislabels            15        10%
Other / random                 35        23%
```

**Step 4: Targeted Interventions**
- Cat/Dog confusion → harder negatives, more diverse training images
- Night image failures → brightness augmentation, adjust normalisation
- Mislabels → manual review and relabelling of suspect examples

This targeted approach is far more effective than blindly trying larger models or more training epochs.

### Phase 6: Writing a Model Card

A **model card** is a short document (1–2 pages) that every serious ML deployment should include. It was introduced by Mitchell et al. (2019) at Google and has become an industry standard.

**Model Card Structure:**
```
MODEL DETAILS
  - Model name and version
  - Architecture: MobileNetV2 + custom head, fine-tuned
  - Training data: CIFAR-10 (50,000 images, 10 classes)
  - Input: 96×96 RGB image, normalised to [-1, 1]
  - Output: 10-class softmax probability vector

INTENDED USE
  - Classify natural images into 10 categories
  - NOT intended for medical diagnosis or safety-critical use

PERFORMANCE METRICS
  - Overall test accuracy: 89.2%
  - Per-class F1 scores: (table)
  - Evaluation dataset: CIFAR-10 test set (10,000 images)

LIMITATIONS
  - Accuracy drops on out-of-distribution images
  - Performance on dark/low-contrast images is lower
  - Bias toward common image compositions

DEPLOYMENT REQUIREMENTS
  - Minimum inference latency: 15 ms on CPU
  - Memory footprint: 14 MB (TFLite, quantised)
```

### Phase 7: Deployment Checklist

Before shipping any model to production, verify:

```
PRE-DEPLOYMENT CHECKLIST
─────────────────────────
✅ Model tested on held-out test set (not used during development)
✅ Preprocessing matches training exactly
✅ Edge cases handled (very dark image, corrupt file, wrong shape)
✅ Latency measured at expected batch size on target hardware
✅ Model versioned and checkpointed
✅ Rollback plan documented
✅ Monitoring metrics defined (accuracy, latency, input distribution)
✅ Model card written and reviewed
✅ Legal/compliance review completed (if applicable)
```

### What Comes Next in Computer Vision

You have mastered image classification. The broader field of computer vision extends far beyond:

**Object Detection** — Not just "what" but "where": predict bounding boxes around multiple objects in a single image. Key architectures: YOLO (real-time), Faster R-CNN (high accuracy), SSD (balanced). Object detection is used in autonomous vehicles, security cameras, medical imaging, and retail analytics.

**Semantic Segmentation** — Predict a class label for every single pixel. Used in autonomous driving (road/car/pedestrian), medical image analysis (tumour boundaries), and satellite image analysis. Key architectures: FCN, U-Net, DeepLab.

**Instance Segmentation** — Like semantic segmentation but distinguishes between individual object instances. Mask R-CNN extends Faster R-CNN with pixel-wise masks.

**Vision Transformers (ViT)** — In 2020, Dosovitskiy et al. showed that the Transformer architecture (originally from NLP) achieves state-of-the-art results on image classification when applied to patches. ViT and its descendants (DeiT, Swin Transformer, CLIP) have transformed computer vision research. Understanding self-attention and positional embeddings from NLP transfers directly.

**Multimodal Models** — Models like CLIP (Contrastive Language-Image Pre-Training) and GPT-4V learn joint representations of images and text, enabling zero-shot classification, image captioning, and visual question answering.

### Recommended Resources for Continuing Education

- **fast.ai Practical Deep Learning for Coders (Part 1 & 2)** — free, practical, project-oriented; excellent complement to this course
- **CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)** — the canonical academic course; lecture notes are excellent reference material
- **"Deep Learning" by Goodfellow, Bengio & Courville** — the comprehensive textbook; Chapters 9–12 are most relevant to this course
- **Papers With Code** (`paperswithcode.com`) — tracks state-of-the-art results with linked code for every major CV benchmark
- **Kaggle Competitions** — practical experience on real datasets with community solutions and notebooks; the most efficient way to develop intuitions about hard problems

### Reflecting on the Full Journey

You started this course understanding pixels as numbers in a matrix. You now understand how hierarchical convolutional feature learning automatically discovers visual patterns from data, how to train and regularise deep networks, how to transfer knowledge from large pretrained models to new tasks, how to interpret what a model has learned, and how to prepare a model for deployment.

The field moves fast, but the foundations you have built — understanding loss landscapes, gradient flow, feature representations, and the bias-variance trade-off — are durable. Every new architecture, whether it is a Vision Transformer, a diffusion model, or something not yet invented, is ultimately built on these same principles.

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| End-to-End Pipeline | The complete sequence from raw data to deployment-ready model | Connects all course concepts into a single coherent workflow |
| Error Analysis | Systematic examination of misclassified examples to identify failure patterns | Higher-leverage than arbitrary model changes; guides targeted improvements |
| Model Card | Structured documentation of a model's design, metrics, limitations, and deployment requirements | Industry standard for responsible AI deployment |
| Hyperparameter Tuning | Systematic search over training configuration to maximise validation performance | Prevents premature convergence to suboptimal configurations |
| Object Detection | Predicting class labels AND bounding box coordinates for multiple objects | Extends classification to spatial localisation; next frontier after this course |
| Semantic Segmentation | Per-pixel class prediction; every pixel assigned a class label | Required for autonomous driving, medical imaging, and precise scene understanding |
| Vision Transformer (ViT) | Transformer architecture applied to image patches instead of tokens | State-of-the-art in large-scale vision; bridges NLP and CV research |
| CLIP | Contrastive Language-Image Pre-Training; joint image+text embedding | Enables zero-shot image classification from natural language descriptions |
| Deployment Checklist | Systematic pre-launch verification of all production requirements | Prevents common production failures: preprocessing mismatch, latency regression |
| Model Versioning | Tracking model iterations with metadata, checkpoints, and rollback capability | Essential for production reliability and iterative improvement |

## Code Reference

See the fully runnable demonstration in **`code/lesson_20.py`**.

## Activities

1. **Full Pipeline Run** — Execute `code/lesson_20.py` end-to-end on CIFAR-10. Record every metric printed: training accuracy, validation accuracy, test accuracy, per-class F1, model size, and inference latency. Annotate the output with your interpretation of each metric.

2. **Error Analysis Report** — After running the capstone code, examine at least 20 misclassified test images. Categorise the errors (e.g. interclass confusion, unusual lighting, ambiguous examples). Estimate what percentage of errors fall into each category and propose one targeted intervention for the largest category.

3. **Write a Model Card** — Using the template in this lesson, write a complete model card for the model trained in Activity 1. Include all sections: model details, intended use, performance metrics, limitations, and deployment requirements.

4. **Grad-CAM Reflection** — Generate Grad-CAM heatmaps for 5 correctly classified and 5 misclassified test images. For each misclassified image, describe where the model was looking and hypothesise why it made the wrong prediction.

5. **Research Extension** — Choose one of the "What Comes Next" topics: object detection, semantic segmentation, or Vision Transformers. Find one paper or tutorial on that topic and write a one-page summary connecting it to what you have learned in this course.

## Review Questions

1. Describe the complete end-to-end pipeline for an image classification project. What is the output of each stage and how does it feed into the next?
2. Explain why error analysis is considered more valuable than blindly scaling up model size. What specific information does it provide that accuracy metrics alone do not?
3. What is a model card and why has it become an industry standard for responsible AI deployment? What consequences can result from deploying a model without one?
4. Compare image classification, object detection, and semantic segmentation: what does each task predict, and how do they differ in output format?
5. Reflecting on the complete course, which concept or technique do you feel most transformed your understanding of how CNNs learn? Explain why.

## Further Reading

- **"Model Cards for Model Reporting"** — Mitchell et al. (2019) — the paper that introduced model cards as a standard practice; required reading for responsible AI practitioners
- **"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"** — Dosovitskiy et al. (2020) — the Vision Transformer paper that sparked the shift from CNNs to transformers in research
- **"CLIP: Learning Transferable Visual Models From Natural Language Supervision"** — Radford et al. (2021) — multimodal learning; demonstrates zero-shot generalisation through language-image alignment
- **fast.ai Practical Deep Learning for Coders** — `https://course.fast.ai` — the best free continuation of this course; project-first, highly practical
- **Kaggle Learn: Computer Vision** — `https://www.kaggle.com/learn/computer-vision` — short, certificate-earning curriculum with hands-on notebooks; excellent for building portfolio projects
