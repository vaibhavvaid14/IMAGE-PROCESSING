import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

print("\n===== INTELLIGENT IMAGE ENHANCEMENT & ANALYSIS SYSTEM =====\n")

os.makedirs("outputs", exist_ok=True)

# TASK 1
print("System initialized")

# TASK 2 : Image Acquisition
image_path = "images/input.webp"
img = cv2.imread(image_path)

if img is None:
    print("Image not found")
    exit()

img = cv2.resize(img, (512,512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/original.png", img)
cv2.imwrite("outputs/grayscale.png", gray)

# TASK 3 : Noise Simulation
gaussian_noise = gray + np.random.normal(0,25,gray.shape)
gaussian_noise = np.clip(gaussian_noise,0,255).astype(np.uint8)

sp_noise = gray.copy()
prob = 0.02
rnd = np.random.rand(*gray.shape)
sp_noise[rnd < prob] = 0
sp_noise[rnd > 1-prob] = 255

cv2.imwrite("outputs/gaussian_noise.png", gaussian_noise)
cv2.imwrite("outputs/salt_pepper.png", sp_noise)

# Restoration
mean_filter = cv2.blur(gaussian_noise,(5,5))
median_filter = cv2.medianBlur(sp_noise,5)
gaussian_filter = cv2.GaussianBlur(gaussian_noise,(5,5),0)

cv2.imwrite("outputs/mean_filter.png", mean_filter)
cv2.imwrite("outputs/median_filter.png", median_filter)
cv2.imwrite("outputs/gaussian_filter.png", gaussian_filter)

# Enhancement
enhanced = cv2.equalizeHist(gaussian_filter)
cv2.imwrite("outputs/enhanced.png", enhanced)

# TASK 4 : Segmentation
_, global_thresh = cv2.threshold(enhanced,127,255,cv2.THRESH_BINARY)
_, otsu_thresh = cv2.threshold(enhanced,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(otsu_thresh,kernel,iterations=1)
dilation = cv2.dilate(erosion,kernel,iterations=1)

cv2.imwrite("outputs/global_threshold.png", global_thresh)
cv2.imwrite("outputs/otsu_threshold.png", otsu_thresh)
cv2.imwrite("outputs/morphology.png", dilation)

# TASK 5 : Edge Detection
sobelx = cv2.Sobel(enhanced,cv2.CV_64F,1,0,ksize=5)
sobelx = cv2.convertScaleAbs(sobelx)

canny = cv2.Canny(enhanced,100,200)

cv2.imwrite("outputs/sobel.png", sobelx)
cv2.imwrite("outputs/canny.png", canny)

# Contours
contours,_ = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

bbox_img = img.copy()

for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    area = cv2.contourArea(c)
    peri = cv2.arcLength(c,True)

    if area > 500:
        cv2.rectangle(bbox_img,(x,y),(x+w,y+h),(0,255,0),2)
        print(f"Area:{area:.2f}  Perimeter:{peri:.2f}")

cv2.imwrite("outputs/bounding_boxes.png", bbox_img)

# Feature Extraction ORB
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(enhanced,None)

feature_img = cv2.drawKeypoints(img,kp,None,color=(0,255,0))
cv2.imwrite("outputs/orb_features.png", feature_img)

# TASK 6 : Evaluation
def mse(a,b):
    return np.mean((a.astype("float")-b.astype("float"))**2)

def psnr(a,b):
    m = mse(a,b)
    if m == 0:
        return 100
    return 20*np.log10(255.0/np.sqrt(m))

mse_val = mse(gray, enhanced)
psnr_val = psnr(gray, enhanced)
ssim_val = ssim(gray, enhanced)

print("\n===== PERFORMANCE METRICS =====")
print("MSE :", mse_val)
print("PSNR:", psnr_val)
print("SSIM:", ssim_val)

# TASK 7 : Final Visualization
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gaussian_noise,cmap="gray")
plt.title("Noisy")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(gaussian_filter,cmap="gray")
plt.title("Restored")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(enhanced,cmap="gray")
plt.title("Enhanced")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(dilation,cmap="gray")
plt.title("Segmented")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(cv2.cvtColor(feature_img,cv2.COLOR_BGR2RGB))
plt.title("Features")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/final_pipeline.png")
plt.show()

print("\nSystem completed successfully")