# Lesson 02: Understanding Digital Images and Pixels

## Learning Objectives
- Explain how a digital image is stored in memory as a multi-dimensional numpy array and describe its shape notation
- Distinguish between grayscale, RGB, and HSV colour spaces and explain when each is used
- Perform pixel-level manipulations including normalisation, channel separation, and colour space conversion
- Describe the concepts of bit depth, image resolution, and common image file formats and their trade-offs
- Interpret an image histogram and use it to reason about contrast and brightness
- Apply normalisation to prepare images for neural network input

---

## Detailed Explanation

### Pixels: The Atoms of Digital Images

The word "pixel" is a portmanteau of "picture element." It is the smallest addressable unit in a digital image. When you look at a digital photograph, you are seeing millions of tiny squares, each filled with a single colour. Zoom in far enough on any digital image and you will see the individual coloured squares.

Each pixel stores one or more numbers. How many numbers, and what range those numbers cover, depends on the colour model and bit depth being used.

### Bit Depth

Bit depth describes how many binary digits (bits) are used to represent a single pixel channel. The most common bit depth in standard image processing and machine learning is **8-bit per channel**, which gives 2⁸ = 256 possible values (0 to 255).

```
1-bit image:    2 values  (pure black or pure white)
4-bit image:   16 values  (limited greyscale or palette)
8-bit image:  256 values  (standard; one byte per channel)
16-bit image: 65536 values (medical/scientific imaging)
32-bit float:  continuous range (used internally in neural nets)
```

Higher bit depth means finer gradations of colour or brightness, but also larger file sizes. Machine learning pipelines almost always work with 8-bit images at the input stage, then convert to 32-bit floating point internally.

### Grayscale Images

A grayscale image stores a single intensity value per pixel. Zero means completely black; 255 means completely white; values in between represent shades of grey. A grayscale image of height H and width W is stored as a 2-dimensional numpy array with shape `(H, W)`.

```
Shape:  (28, 28)   ← MNIST digits are 28×28 grayscale
Memory: 28 × 28 × 1 byte = 784 bytes per image
```

Internally, the intensity value represents luminance — how much light a surface appears to emit or reflect. Grayscale images discard colour information, which is often acceptable for tasks where colour is irrelevant (e.g., digit recognition, X-ray analysis, document processing).

### RGB Colour Images

The RGB model represents colour using three additive primary channels: **Red**, **Green**, and **Blue**. Each channel is an 8-bit grayscale image (0–255). The three channels are combined — conceptually "stacked" — to produce the full-colour image.

```
RGB Image Shape:  (H, W, 3)
   Channel 0: Red
   Channel 1: Green
   Channel 2: Blue

CIFAR-10 example:  (32, 32, 3)
Memory: 32 × 32 × 3 bytes = 3,072 bytes per image
```

When R=255, G=0, B=0, the pixel is pure red. When R=255, G=255, B=0, the pixel is yellow (red + green = yellow in additive colour mixing). When R=G=B, the pixel is a shade of grey.

```
ASCII diagram of RGB channels for a small image patch:

Original colour pixel (orange):  R=255, G=165, B=0

Red channel:    [ 255 ]   (bright — orange has lots of red)
Green channel:  [ 165 ]   (medium — orange has some green)
Blue channel:   [   0 ]   (dark — orange has no blue)
```

### HSV Colour Space

RGB is intuitive for computers but not for humans. We tend to think about colour in terms of hue (what colour?), saturation (how vivid?), and value (how bright?). The HSV model encodes exactly these three qualities:

- **Hue (H):** The "pure colour," represented as an angle on a colour wheel (0°–360°, or 0–179 in OpenCV's 8-bit representation)
- **Saturation (S):** Purity of the colour. 0 = grey, 255 = fully saturated pure colour
- **Value (V):** Brightness. 0 = black, 255 = maximum brightness

HSV is particularly useful when you want to isolate objects by their colour. For example, to detect ripe tomatoes (red) in an image, it is far easier to threshold on the Hue channel than to reason about all the combinations of R, G, B values that together produce "red."

### How Images Are Stored in Memory as Numpy Arrays

NumPy arrays are the lingua franca of image data in Python. When you load an image, it becomes a numpy array immediately.

```
Grayscale:   array shape  (H, W)         dtype: uint8
Colour:      array shape  (H, W, 3)      dtype: uint8
Batch:       array shape  (N, H, W, C)   dtype: float32 after normalisation

Where:
  N = number of images (batch size)
  H = image height in pixels
  W = image width in pixels
  C = number of channels (1 for grayscale, 3 for RGB)
```

Understanding array shape is critical because every layer of a neural network expects input in a specific shape. Keras CNNs expect inputs shaped `(N, H, W, C)` — this is called "channels last" format. The alternative "channels first" format `(N, C, H, W)` is used by some other frameworks (PyTorch defaults to this).

### Normalisation: Dividing by 255

Raw pixel values range from 0 to 255. Neural networks are sensitive to the scale of their inputs. Gradient descent optimisation works poorly — and often diverges entirely — when input values are large and on wildly different scales.

The simplest normalisation is to divide every pixel value by 255.0, mapping the range [0, 255] to [0.0, 1.0]:

```
normalised_pixel = raw_pixel / 255.0

Example:
  Raw pixel value:         200
  Normalised value:        200 / 255.0 ≈ 0.784
```

This is called **min-max normalisation** or **[0, 1] scaling**. Another common approach is **standardisation** (subtracting the mean and dividing by the standard deviation), which maps values to a distribution centred at 0 with standard deviation 1 — sometimes called z-score normalisation. ImageNet pre-trained models typically use per-channel mean subtraction with specific mean values (e.g., mean=[0.485, 0.456, 0.406] for ImageNet).

For MNIST and CIFAR-10 in this course, simple division by 255.0 is sufficient.

### Image Histograms

A histogram of pixel values shows the distribution of intensities across an image. The x-axis represents pixel values (0–255), and the y-axis represents how many pixels in the image have each value.

```
Dark image histogram:         Bright image histogram:
  Counts                         Counts
    │█                              │          █
    │███                            │        █████
    │█████                          │      ████████
    └──────────────── Values        └──────────────── Values
    0                255            0                 255
    (peaks on left)                 (peaks on right)
```

Histograms reveal whether an image is:
- **Underexposed** (too dark): histogram concentrated near 0
- **Overexposed** (too bright): histogram concentrated near 255
- **Low contrast**: narrow histogram clustered in the middle
- **High contrast**: histogram spread across the full range

In machine learning, histograms of the training set's pixel distribution can reveal dataset biases — for example, if nighttime images dominate, the model may struggle with well-lit scenes.

### Common Image Formats

| Format | Compression | Channels | Common Use |
|---|---|---|---|
| **JPEG** | Lossy | RGB | Photographs, web images |
| **PNG** | Lossless | RGB + Alpha | Screenshots, graphics with transparency |
| **GIF** | Lossless | Palette (256 colours) | Animations, simple graphics |
| **BMP** | None (uncompressed) | RGB | Windows native format |
| **TIFF** | Lossless optional | Multi-channel | Medical, scientific, professional |
| **WEBP** | Lossy or lossless | RGB + Alpha | Modern web format |

JPEG compression discards fine detail that the human eye has difficulty perceiving, achieving file sizes 10× smaller than uncompressed formats. However, each time a JPEG is saved, it loses more data — never save intermediate processing results as JPEG. Use PNG or NPY (numpy format) instead.

### Image Resolution

Resolution refers to the number of pixels in an image and directly determines its level of detail. A 4K image (3840 × 2160) has over 8 million pixels. MNIST digits are 28 × 28 = 784 pixels. CIFAR-10 images are 32 × 32 = 1,024 pixels.

Higher resolution captures more detail but requires more memory and computation. Most CNN architectures expect inputs of a fixed size (224×224 for ImageNet models, 299×299 for Inception). Resizing an image to fit the model's expected input is one of the first pre-processing steps.

### Practical Misconception: "The Colour Values Mean Something to the Network"

A neural network has no concept of "red" or "blue." It sees the number 255 in channel 0 of a pixel. The meaning of that number — that it corresponds to red in the RGB model — is entirely a human convention. The network learns correlations between numerical patterns and labels. If you swapped R and B channels for every image in your training set and did the same at test time, the network would learn to classify just as well.

---

## Key Concepts

| Term | Definition | Why It Matters |
|---|---|---|
| **Pixel** | The smallest unit of a digital image, storing one or more numerical intensity values | All image processing starts and ends with pixel values |
| **Bit Depth** | Number of bits used to encode each channel per pixel; 8-bit gives 256 levels | Determines image fidelity and file size |
| **Grayscale** | Single-channel image encoding luminance only, shape (H, W) | Simpler than colour; used for X-rays, handwriting, documents |
| **RGB** | Three-channel colour model (Red, Green, Blue), shape (H, W, 3) | The dominant colour representation for camera images |
| **HSV** | Hue-Saturation-Value colour space; separates colour from brightness | Easier for colour-based object segmentation and detection |
| **Normalisation** | Scaling pixel values from [0, 255] to [0, 1] by dividing by 255 | Ensures stable gradient descent during neural network training |
| **Numpy Array** | Multi-dimensional array that holds image pixel data in Python | The universal data structure for images in Python ML pipelines |
| **Image Histogram** | Count of pixels at each intensity level | Reveals image quality, brightness, and contrast characteristics |
| **Channels Last** | Tensor format (N, H, W, C) used by TensorFlow/Keras | Knowing axis order prevents shape mismatch errors |

---

## Code Reference

See `code/lesson_02.py` for runnable demos: loading and manipulating pixel arrays, converting between colour spaces, separating channels, normalising images, and plotting histograms.

---

## Activities

1. **Shape Investigation:** Load both an MNIST image and a CIFAR-10 image in `lesson_02.py`. Print the shape, dtype, minimum value, maximum value, and mean value for each. Explain in a comment why the shapes differ.

2. **Channel Surgery:** For a CIFAR-10 colour image, extract the three individual channels (R, G, B) into separate arrays. Set the R channel to zero (creating a cyan-tinted image) and display the result alongside the original. Describe how the colours change.

3. **Histogram Analysis:** Plot pixel-value histograms for three different CIFAR-10 classes (e.g., "automobile," "frog," "airplane"). Do the histograms look different? What does that tell you about the colour characteristics of each class?

4. **Normalisation Verification:** Before and after dividing by 255.0, print the min, max, and mean pixel values. Verify mathematically that `max_value / 255.0 = 1.0` and `min_value / 255.0 = 0.0`. Try using standardisation (subtract mean, divide by std) and compare the resulting value range.

5. **Format Experiment:** Save an MNIST digit as a JPEG, then reload it. Compare the pixel values of the original numpy array and the reloaded JPEG. Are they identical? Calculate the maximum absolute difference. This demonstrates JPEG's lossy compression.

---

## Review Questions

1. A colour image has shape `(480, 640, 3)`. What do each of the three dimensions represent? How much memory does this image consume if each pixel channel is stored as a uint8?

2. Explain why neural networks require normalised pixel values in the range [0, 1] rather than raw integer values in [0, 255]. What problem does large-scale input cause during training?

3. Describe a real-world scenario where the HSV colour space would be more useful than RGB for a computer vision task. Explain your reasoning step by step.

4. What is the difference between lossy and lossless image compression? Name one format of each type and explain the trade-off in the context of a machine learning dataset.

5. If you have a batch of 64 grayscale images each 28×28 pixels, what shape should the numpy array be for input to a Keras CNN? Write the shape notation and explain what each dimension represents.

---

## Further Reading

- **NumPy documentation: Array indexing and slicing** — Understanding how to slice multi-dimensional arrays is essential for all image manipulation work in Python.
- **"Digital Image Processing" by Gonzalez and Woods (Chapters 1–3)** — The classic textbook on image fundamentals; Chapters 1–3 cover pixels, colour spaces, and histograms at depth.
- **OpenCV documentation: Colour Space Conversions** — Official guide to all colour space transforms available in OpenCV with code examples.
- **Keras documentation: image_dataset_from_directory** — Learn how real-world image datasets are loaded, resized, and normalised for deep learning pipelines.
- **"Understanding JPEG Compression" — Computerphile YouTube channel** — Excellent visual explanation of how JPEG compression works and why it is lossy.
