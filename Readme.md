# 📄 Smart Document Scanner & Quality Analysis System

## Image Processing & Computer Vision — Mini Project

**Student Name:** Lakshay 
**Roll No:** 2301010333
**Course:** B.Tech CSE
**Assignment:** Smart Document Scanner using Python
**Submission Date:** 13-04-2026

---

## 🧾 Introduction

Nowadays almost every organization converts physical documents into digital format. Universities scan answer sheets, banks scan forms, and offices digitize paperwork for storage and processing.

But scanning is not just about taking a picture. The **quality of the captured image** directly affects readability and OCR performance. Poor resolution or low gray levels can make text unclear or even unreadable.

In this mini project, I created a simple **Smart Document Scanner simulation** using Python to understand how image quality changes when resolution and gray levels are reduced.

The goal was not to build a commercial scanner, but to practically observe how **sampling** and **quantization** influence document clarity.

---

## 🎯 Objectives of the Project

* Understand how document images are acquired.
* Convert colored documents into grayscale for processing.
* Study the effect of image resolution (sampling).
* Study the effect of gray-level reduction (quantization).
* Compare document readability visually.
* Analyze suitability of images for OCR systems.

---

## 🛠️ Tools & Technologies Used

* Python 3
* OpenCV (Image Processing)
* NumPy (Array Operations)
* Matplotlib (Visualization)

---

## 📁 Project Folder Structure

```
document_scanner/
│
├── scanner.py
├── images/
│   ├── doc1.jpg
│   ├── doc2.jpg
│   └── doc3.jpg
│
├── outputs/
│   ├── sample_512.png
│   ├── sample_256.png
│   ├── sample_128.png
│   ├── quant_8bit.png
│   ├── quant_4bit.png
│   ├── quant_2bit.png
│   └── final_comparison.png
│
└── README.md
```

---

## ⚙️ Working of the System

### 1. Image Acquisition

A document image is loaded from the images folder.
The image is resized to **512 × 512** so all comparisons remain consistent.
Then the image is converted into grayscale since most document processing systems work on grayscale data.

---

### 2. Sampling (Resolution Analysis)

To simulate real scanners with different quality levels, the image resolution was changed:

* **512×512** → High quality scan
* **256×256** → Medium quality scan
* **128×128** → Low quality scan

After reducing resolution, images were upscaled again for visualization so differences could be clearly observed.

---

### 3. Quantization (Gray-Level Reduction)

Next, gray levels were reduced to study intensity loss:

* **8-bit (256 levels)** → Original quality
* **4-bit (16 levels)** → Reduced intensity precision
* **2-bit (4 levels)** → Extreme information loss

This step demonstrates how limited bit depth affects visual perception and text readability.

---

### 4. Quality Analysis

All generated images were displayed together in one comparison figure.
This allowed direct visual inspection of:

* Edge sharpness
* Text clarity
* Noise and banding artifacts
* Overall readability

---

## ▶️ How to Run the Project

### Install Required Libraries

```
pip install opencv-python numpy matplotlib
```

### Add Document Images

Place at least three document images inside:

```
images/
```

Examples used:

* Printed notes
* Scanned book page
* Mobile captured document

### Run Program

```
python scanner.py
```

All processed results will automatically appear inside the **outputs/** folder.

---

## 📊 Observations

### Sampling Results

* High resolution images preserve fine text details.
* Medium resolution slightly reduces sharpness.
* Low resolution causes characters to merge and become blurry.

### Quantization Results

* 8-bit images appear natural and readable.
* 4-bit images show visible intensity steps.
* 2-bit images lose most document information.

### OCR Suitability

* Best OCR accuracy occurs with high resolution and higher gray levels.
* Extreme resolution reduction or quantization significantly degrades recognition quality.

---

## 🌍 Real-World Applications

* Mobile scanning applications
* Digital document archiving
* Banking verification systems
* Automated OCR pipelines
* Academic paper digitization

---

## ✅ Assignment Requirements Covered

* Project setup and structured repository
* Image acquisition and preprocessing
* Sampling analysis
* Quantization analysis
* Single comparison visualization
* Output image saving
* Observation-based analysis
* Clean and readable implementation

---

## 📚 References

* OpenCV Official Documentation
* NumPy Documentation
* Matplotlib Documentation
* Course Lecture Notes on Image Processing

---

## ⚖️ Academic Integrity Statement

This project was completed individually as part of coursework.
All implementations were written independently while using online documentation only for conceptual understanding.

---

## 👨‍💻 Author

Vaibhav Vaid
B.Tech Computer Science Engineering