"""
Name: VAibhav VAid
Roll No: 2301010289
Course: Image Processing & Computer Vision
Unit: Document Image Analysis
Assignment: Smart Document Scanner & Quality Analysis
Date: 14 Apr 2026
"""

print("Welcome to Smart Document Scanner System")
print("Simulating Sampling and Quantization Effects...")

# Task 2

import cv2
import matplotlib.pyplot as plt

image = cv2.imread("images/printed.jpg")
image = cv2.resize(image, (512, 512))

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

plt.subplot(1,2,2)
plt.title("Grayscale")
plt.imshow(gray, cmap='gray')

plt.show()



# Task 3


def sample_image(img, size):
    small = cv2.resize(img, (size, size))
    upscaled = cv2.resize(small, (512,512))
    return upscaled

high = gray
medium = sample_image(gray, 256)
low = sample_image(gray, 128)


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("512x512")
plt.imshow(high, cmap='gray')

plt.subplot(1,3,2)
plt.title("256x256")
plt.imshow(medium, cmap='gray')

plt.subplot(1,3,3)
plt.title("128x128")
plt.imshow(low, cmap='gray')

plt.show()

cv2.imwrite("outputs2/sample_512.png", high)
cv2.imwrite("outputs2/sample_256.png", medium)
cv2.imwrite("outputs2/sample_128.png", low)



# Task 4

import numpy as np

def quantize(img, levels):
    step = 256 // levels
    quantized = (img // step) * step
    return quantized.astype(np.uint8)

q8 = quantize(gray, 256)   # 8-bit
q4 = quantize(gray, 16)    # 4-bit
q2 = quantize(gray, 4)     # 2-bit


plt.figure(figsize=(10,4))

plt.subplot(1,3,1)
plt.title("8-bit")
plt.imshow(q8, cmap='gray')

plt.subplot(1,3,2)
plt.title("4-bit")
plt.imshow(q4, cmap='gray')

plt.subplot(1,3,3)
plt.title("2-bit")
plt.imshow(q2, cmap='gray')

plt.show()

cv2.imwrite("outputs2/quant_8bit.png", q8)
cv2.imwrite("outputs2/quant_4bit.png", q4)
cv2.imwrite("outputs2/quant_2bit.png", q2)


# Task 5

plt.figure(figsize=(12,8))

images = [gray, high, medium, low, q8, q4, q2]
titles = [
"Original Gray",
"512x512",
"256x256",
"128x128",
"8-bit",
"4-bit",
"2-bit"
]

for i in range(len(images)):
    plt.subplot(3,3,i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis("off")

plt.savefig("outputs/final_comparison.png")
plt.show()


print("\n--- Quality Analysis ---")

print("1. High resolution retains sharp edges.")
print("2. Low resolution causes text blur and loss of fine characters.")
print("3. Lower gray levels introduce banding artifacts.")
print("4. OCR works best on high resolution and higher bit-depth images.")
print("5. Extremely low quantization reduces readability.")