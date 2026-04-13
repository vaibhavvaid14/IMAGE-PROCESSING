
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n===== MEDICAL IMAGE COMPRESSION & SEGMENTATION SYSTEM =====\n")

os.makedirs("outputs", exist_ok=True)

# TASK 1

image_path = "images/xrayy.jpg"

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print("Image not found!")
    exit()

cv2.imwrite("outputs/original.png", img)
print("Medical image loaded.")


def rle_encode(image):
    flat = image.flatten()
    encoded = []

    prev_pixel = flat[0]
    count = 1

    for pixel in flat[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoded.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1

    encoded.append((prev_pixel, count))
    return encoded


print("\nPerforming RLE Compression...")

rle_data = rle_encode(img)

original_size = img.size
compressed_size = len(rle_data) * 2

compression_ratio = original_size / compressed_size
storage_saving = (1 - (compressed_size / original_size)) * 100

print(f"Original Pixels : {original_size}")
print(f"Compressed Units: {compressed_size}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Storage Saving : {storage_saving:.2f}%")

with open("outputs/rle_output.txt", "w") as f:
    f.write(str(rle_data))


# TASK 2

print("\nApplying Global Thresholding...")

_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("outputs/global_threshold.png", global_thresh)

print("Applying Otsu Thresholding...")

_, otsu_thresh = cv2.threshold(
    img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite("outputs/otsu_threshold.png", otsu_thresh)


# TASK 3

print("\nApplying Morphological Processing...")

kernel = np.ones((3, 3), np.uint8)

erosion = cv2.erode(otsu_thresh, kernel, iterations=1)
cv2.imwrite("outputs/erosion.png", erosion)

dilation = cv2.dilate(erosion, kernel, iterations=1)
cv2.imwrite("outputs/dilation.png", dilation)


# TASK 4

print("\n===== SEGMENTATION ANALYSIS =====")

print("Global thresholding segments image using fixed intensity.")
print("Otsu’s method automatically selects optimal threshold.")
print("Morphological operations remove noise and refine regions.")
print("Refined regions may represent bones, organs, or abnormal tissues.")


plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_thresh, cmap='gray')
plt.title("Global Threshold")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(otsu_thresh, cmap='gray')
plt.title("Otsu Threshold")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation (Refined)")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/final_comparison.png")
plt.show()

print("\nOutputs saved successfully in outputs/ folder.")
print("Project completed successfully!")

