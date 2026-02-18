# CNN-based Image Denoiser
*High-Performance Local-Context Neural Image Restoration Model*

<div align="center">
  <a href="https://cnn-image-denoiser.onrender.com">
    <b>Live demo is available here</a>
    <br>
  ❗️ Note: try to use images lower than 512x512 to prevent memory overrun
    </b>
</div>

---

## Project Overview

**CNN-based Image Denoiser** is a neural network model for image denoising based on **convolutional neural networks (CNNs)**.
The model efficiently restores images using **local patterns and textures**, making it ideal for **real-time and embedded systems** where speed and stability are critical.

The project demonstrates a classic **low-level computer vision** approach, comparable to traditional filters and modern transformers.

⚠️ This implementation is a demonstration of convolutional neural networks and does not currently represent a universal solution for defect removal. For different types of photos with artifacts of varying structure a good result is not guaranteed.

---

## Processing examples

**The example photos were processed in two passes. Network parameters:**
* feature channels = 64
* kernel size = 3
* layers = 5
* dilation = [1, 1, 1, 2, 1, 1, 1]
* image size = 1024

<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_003_layers%3D3_channels%3D48.jpg" width="500"><br>
  <em>Noise between bright and dark areas</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_001_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Local cleaning of noisy areas</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_002_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>ISO noise and grain reduction</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_004_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Removing defects and restoring the picture</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_005_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Smoothing details from point-and-shoot cameras</em>
</p>
<p align="center">
  <img src="https://github.com/lovk4ch/cnn_denoiser/blob/master/data/demo/images/ex_006_layers%3D5_channels%3D64.jpg" width="500"><br>
  <em>Removing fine roughness</em>
</p>

---

## Objective

* Additive noise removal (Gaussian/synthetic noise)
* Local detail restoration

Comparison with:
* Classic filters (Gaussian, Median)
* Baseline (Blur Filters)

---

## Model Architecture

```
Input Image
   ↓
Convolution Block × N
   ↓
Residual Connections
   ↓
BatchNorm + ReLU
   ↓
Final Convolution
   ↓
Denoised Image
```

---

## Key Components

* **3x3 / 5x5 convolution layers**
* **Residual blocks** for a stable gradients
* **ReLU / LeakyReLU activations**
* **Batch Normalization**

---

## Advantages over Blur Filters

### ❌ Classical Blur Filters

* Gaussian / Median / Bilateral
* Works only locally
* Blurs the boundaries
* Doesn't adapt to the scene structure

### 🚀 Key Advantages

* **Preserves local structures and textures**
* **Adaptable to different noise levels**
* **Fast inference suitable for embedded systems**

### ✅ CNN-based Denoiser

| Feature                       | Blur Filter | CNN |
| ----------------------------- | ----------- | --- |
| Local pattern recognition     | ⚡           | ✅   |
| Edge / structure preservation | ❌           | ✅   |
| Adaptive learning             | ❌           | ✅   |
| Real-time inference           | ✅           | ⚡   |
| Training on dataset           | ❌           | ✅   |

### ⚠️ Trade-offs
* Limited global context (patch-to-patch long-range)
* Doesn't always perfectly reconstruct complex noise patterns

---

## 📊 Evaluation Metrics

### Image Quality

* **PSNR** — Peak Signal-to-Noise Ratio
* **SSIM** — Structural Similarity Index

---

## Dataset

* Synthetic noisy DSLR-images

## Noise Model
* Gaussian Noise (σ configurable)
* Luminance ISO noise
* Chroma noise
* Various mixes of all types

---

## Tech Stack

* 🐍 Python
* 🔥 PyTorch
* 🖥️ Custom CNN architecture
* 📊 TensorBoard

---

## Key Takeaways

* ✔️ Demonstrates **local-pattern-based denoising**
* ✔️ Shows **practical use of CNNs in applied CV**
* ✔️ Provides **baseline for comparison with transformer-based denoisers**
* ✔️ Relevant for **Applied ML / Computer Vision roles**

---

## 🔀 Next Releases:

* Add attention layers for extended context (CNN + Attention)
* Move towards diffusion-based denoising
* Add FiLM filters to adjust the level and type of noise
* Add transformed-based restoration models net for the next improvement of out images
