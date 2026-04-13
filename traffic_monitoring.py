
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("\n===== FEATURE BASED TRAFFIC MONITORING SYSTEM =====\n")

os.makedirs("outputs", exist_ok=True)

# TASK 1

image_path = "images/traffic.webp"

img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", img)

print("Applying Sobel Edge Detection")

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = cv2.convertScaleAbs(sobel)

cv2.imwrite("outputs/sobel.png", sobel)

print("Applying Canny Edge Detection")

canny = cv2.Canny(gray, 100, 200)
cv2.imwrite("outputs/canny.png", canny)

# TASK 2

print("\nDetecting Contours")

contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    if area > 500:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x+w, y+h), (0,255,0), 2)
        perimeter = cv2.arcLength(cnt, True)
        print(f"Object Area: {area:.2f}, Perimeter: {perimeter:.2f}")

cv2.imwrite("outputs/contours.png", contour_img)

# TASK 3

print("\nExtracting ORB Features")

orb = cv2.ORB_create(nfeatures=500)

keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(
    img,
    keypoints,
    None,
    color=(0,255,0),
    flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS
)

cv2.imwrite("outputs/orb_features.png", feature_img)

print(f"Total Keypoints Detected: {len(keypoints)}")

# TASK 4

print("\n===== COMPARATIVE ANALYSIS =====")
print("Sobel detects gradient intensity.")
print("Canny produces sharper edges.")
print("Contours represent vehicles and road objects.")
print("ORB keypoints help in tracking vehicles and motion analysis.")

plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(sobel, cmap='gray')
plt.title("Sobel")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(canny, cmap='gray')
plt.title("Canny")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.title("Contours")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
plt.title("ORB Features")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/final_comparison.png")
plt.show()

print("\nOutputs saved in outputs folder")
print("Project Completed Successfully")
