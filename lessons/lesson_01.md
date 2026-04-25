# Lesson 01: What Is Image Classification?

## Learning Objectives
- Define image classification and distinguish it from related tasks such as object detection and image segmentation
- Identify at least five real-world domains where image classification is actively deployed
- Describe how a computer represents and "sees" an image as a matrix of numerical values
- Trace the full image classification pipeline from raw pixel input through feature extraction to final prediction
- Explain the historical shift from hand-crafted feature engineering to end-to-end deep learning
- Articulate why deep learning now dominates image classification benchmarks

---

## Detailed Explanation

### What Is Image Classification?

Image classification is the task of assigning a single label — chosen from a fixed set of categories — to an entire image. Given a photograph of an animal, an image classifier returns an answer like "cat," "dog," or "bird." That is the whole job: one image in, one class label out.

This sounds deceptively simple, but it is one of the hardest problems in computer science because the same real-world object can look dramatically different depending on lighting, angle, occlusion, scale, and background clutter. A coffee mug looks entirely different when photographed from above versus from the side, yet a human — and a well-trained model — recognises it instantly in both cases.

### Distinguishing Classification from Related Tasks

It is important to understand what image classification is *not*:

| Task | Output | Example |
|---|---|---|
| **Image Classification** | One label per image | "This image contains a dog" |
| **Object Detection** | Bounding boxes + labels | "Dog at top-left, cat at bottom-right" |
| **Semantic Segmentation** | Per-pixel class label | Every pixel coloured by class |
| **Instance Segmentation** | Per-pixel + individual instances | Two dogs, each outlined separately |
| **Image Captioning** | Natural language sentence | "A dog running on a beach" |

This course focuses on classification — the foundation upon which all other vision tasks are built.

### How Computers "See" Images

Humans perceive images holistically. Computers do not. To a computer, an image is nothing more than a rectangular grid of numbers. A grayscale image that is 28 pixels tall and 28 pixels wide is stored as a 28 × 28 matrix where each cell holds a single integer from 0 (black) to 255 (white). A colour image adds a third dimension: three such matrices stacked together, one for red, one for green, and one for blue — giving a shape of 28 × 28 × 3.

```
Grayscale 5×5 image (conceptual):
  0   0   0   0   0
  0 128 255 128   0
  0 255 255 255   0
  0 128 255 128   0
  0   0   0   0   0
        ↑
  Pixel value 255 = white
  Pixel value   0 = black
```

There is no inherent meaning in these numbers to the computer. All of the "meaning" — the idea that a cluster of certain pixel values represents an eye, and a certain arrangement of eyes and a nose represents a face — must be *learned* from data.

### The Classification Pipeline

Every image classification system, whether traditional or deep learning-based, follows the same conceptual pipeline:

```
┌─────────────┐    ┌──────────────────┐    ┌───────────────┐    ┌────────────┐
│  Raw Image  │───▶│  Pre-processing  │───▶│   Feature     │───▶│ Prediction │
│ (pixels)    │    │  (resize,        │    │  Extraction   │    │ (class     │
│             │    │   normalise)     │    │               │    │  label)    │
└─────────────┘    └──────────────────┘    └───────────────┘    └────────────┘
```

**Step 1 — Input:** The raw image is loaded as a pixel array.

**Step 2 — Pre-processing:** The image is resized to a standard dimension (e.g., 224 × 224), pixel values are normalised (usually divided by 255 so they fall in the range [0, 1]), and data augmentation may be applied during training.

**Step 3 — Feature Extraction:** This is the heart of the system. In traditional computer vision, engineers manually designed features such as edge histograms, texture descriptors, and colour histograms. In deep learning, a Convolutional Neural Network (CNN) learns these features automatically from data by adjusting millions of numerical parameters.

**Step 4 — Prediction:** The extracted features are passed to a classifier (a softmax layer in deep learning) that outputs a probability for each class. The class with the highest probability is the prediction.

### Real-World Applications

Image classification powers a remarkable range of technologies:

**Medical Imaging:** Convolutional networks classify chest X-rays as showing pneumonia or not, detect diabetic retinopathy in retinal scans with accuracy matching specialist doctors, and identify cancerous cells in pathology slides. Early detection saves lives, and AI can process thousands of scans per hour.

**Self-Driving Vehicles:** Cameras mounted on autonomous cars classify traffic signs, road markings, and pedestrian crossing signals in real time, hundreds of times per second. A misclassification can have life-or-death consequences, making accuracy and speed both critical.

**Content Moderation:** Social media platforms classify billions of images per day to detect nudity, graphic violence, hate symbols, and misinformation. Manual review at this scale is impossible; automated classification makes it tractable.

**Agriculture:** Drones capture aerial images of crops, and classifiers identify patches affected by disease, drought, or pest infestation, allowing targeted treatment rather than blanket pesticide application.

**Manufacturing Quality Control:** Cameras on assembly lines classify products as defective or acceptable at production speed, replacing tedious manual inspection.

**Retail and E-Commerce:** Visual search tools let customers photograph a product they like and find similar items in a catalogue. Classification drives product recommendations and automatic catalogue tagging.

### A Brief History: From Hand-Crafted Features to Deep Learning

For decades, progress in image recognition required domain experts who manually designed features — complex algorithms to extract edges (Canny, Sobel), interest points (SIFT, SURF), and texture patterns (LBP, HOG). A typical pipeline in the early 2000s looked like:

```
Image → HOG Features → SVM Classifier → Label
```

These systems worked reasonably well on constrained datasets but required enormous engineering effort and failed to generalise to complex real-world scenes.

The turning point came in 2012 when Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton published AlexNet, a deep CNN that won the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) by a margin of roughly 10 percentage points over the second-place entry. AlexNet demonstrated that deep neural networks trained on large datasets with GPUs could learn better feature extractors than anything engineers had designed by hand.

Since then, a rapid succession of architectures — VGGNet (2014), GoogLeNet/Inception (2014), ResNet (2015), DenseNet (2016), EfficientNet (2019) — have pushed classification accuracy on ImageNet from around 72% (AlexNet) to over 90% (EfficientNet-L2). Some benchmarks now show superhuman performance.

### Why Does Deep Learning Win?

Three factors explain deep learning's dominance:

1. **Scale of data:** ImageNet contains 1.2 million labelled images across 1,000 classes. Deep networks have enough parameters to represent extremely complex patterns, but they need vast amounts of data to learn those patterns without overfitting.

2. **Computational power:** Modern GPUs and TPUs can perform the billions of floating-point multiplications required to train a large CNN in hours rather than years.

3. **End-to-end learning:** Deep networks learn the *entire* pipeline — from raw pixels to final prediction — jointly. There is no hand-crafted intermediate representation, so no human bias constrains what features the network can discover.

### Common Misconceptions

**"The network understands what it sees."** Neural networks do not understand images in any human sense. They learn statistical correlations between pixel patterns and labels. A network trained to classify wolves vs. huskies once learned to detect snow in the background rather than the animal, because training wolves were always photographed in snowy environments.

**"More data always helps."** Data quality matters more than quantity. A million mislabelled images can be worse than 100,000 correctly labelled ones.

**"Image classification and object recognition are the same thing."** Classification assigns one label to the whole image. Recognition (detection) finds and locates multiple objects within an image. They use related but distinct architectures.

---

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| **Image Classification** | Assigning a single label from a predefined set to an entire input image | The foundational task in computer vision; underpins most vision applications |
| **Pixel** | The smallest discrete element of a digital image, holding one or more numerical values | Images are entirely composed of pixels; understanding pixel values is the starting point |
| **Feature Extraction** | The process of transforming raw pixel data into a compact, meaningful representation | Determines whether a classifier can distinguish between classes |
| **Pipeline** | The sequence of processing steps from raw input to final prediction | Understanding the pipeline reveals where errors can occur and how to fix them |
| **Object Detection** | Locating and labelling individual objects within an image using bounding boxes | Contrasts with classification; highlights the scope difference |
| **Softmax** | A mathematical function that converts a vector of raw scores into a probability distribution | The standard final layer for multi-class classification |
| **ImageNet** | A dataset of ~14 million labelled images across 20,000+ categories | The benchmark that defined modern deep learning for vision |
| **Overfitting** | When a model performs well on training data but poorly on unseen data | A central challenge in training any machine learning model |

---

## Code Reference

See `code/lesson_01.py` for runnable demos: loading and displaying MNIST images, exploring pixel arrays, and visualising class distribution.

---

## Activities

1. **Pixel Explorer:** Run `code/lesson_01.py` and examine the printed pixel array for a handwritten digit. Change the index to display 10 different digits. Record how the pixel values differ between a "1" (thin stroke) and an "8" (dense strokes).

2. **Class Distribution Chart:** Modify the histogram section of `lesson_01.py` to display class distribution for only the *test* set instead of the training set. Compare the distributions — are they balanced?

3. **Manual Brightness Change:** After loading an MNIST image (a 28×28 numpy array), write code to multiply every pixel value by 0.5 (darken) and display the result. Then try multiplying by 1.5 (brighten, then clip at 255). Describe what you observe.

4. **Classification vs. Detection Research:** Find one news article or paper describing a real-world image *detection* system (e.g., YOLO, Faster R-CNN). Write a one-paragraph summary explaining what makes it different from classification.

5. **Build a Mental Pipeline:** Draw your own version of the classification pipeline diagram (on paper or digitally) for the specific domain of classifying skin lesion photographs as benign or malignant. Label each stage with domain-specific details (e.g., "pre-processing: standardise to 224×224, adjust for skin tone bias").

---

## Review Questions

1. Explain in your own words why a computer cannot "see" an image the way a human does, and describe what a computer actually stores when it loads an image file.

2. What are three real-world applications of image classification, and for each one, explain what the consequences of a misclassification might be.

3. Describe the four stages of the image classification pipeline and explain what happens at each stage.

4. What was the significance of AlexNet in 2012, and what three factors does the text identify as explaining deep learning's dominance over traditional computer vision?

5. Explain the difference between image classification, object detection, and semantic segmentation using an original example (not one from the lesson).

---

## Further Reading

- **"ImageNet Large Scale Visual Recognition Challenge" (Russakovsky et al., 2015)** — The foundational paper describing the benchmark that drove modern deep learning; freely available on arXiv.
- **"Deep Learning" by Goodfellow, Bengio, and Courville (Chapter 9: Convolutional Networks)** — A rigorous yet accessible introduction to the mathematics behind CNNs.
- **cs231n.stanford.edu (Stanford CS231n: Convolutional Neural Networks for Visual Recognition)** — Free lecture notes and slides covering the entire journey from pixels to state-of-the-art architectures.
- **"How computers learn to recognize objects instantly" — TED Talk by Joseph Redmon** — A compelling 15-minute accessible overview of real-time object detection (YOLO), providing great context for why classification is step one.
- **Keras documentation: datasets module** — Official documentation for the built-in datasets (MNIST, CIFAR-10, etc.) used throughout this course; great for understanding data shapes and conventions.
