
import cv2
import matplotlib.pyplot as plt
import numpy as np
# Task 1

img = cv2.imread("images/street.webp")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.show()

# task 2

noise = np.random.normal(0, 25, gray.shape)
gaussian_noisy = gray + noise
gaussian_noisy = np.clip(gaussian_noisy,0,255).astype(np.uint8)

cv2.imwrite("outputs/gaussian_noise.png", gaussian_noisy)



sp_noisy = gray.copy()

prob = 0.02
rand = np.random.rand(*gray.shape)

sp_noisy[rand < prob/2] = 0
sp_noisy[rand > 1 - prob/2] = 255


cv2.imwrite("outputs/salt_pepper_noise.png", sp_noisy)
#task 3

mean_filtered = cv2.blur(gaussian_noisy, (5,5))

cv2.imwrite("outputs/mean_filter.png", mean_filtered)

median_filtered = cv2.medianBlur(sp_noisy,5)

cv2.imwrite("outputs/median_filter.png", median_filtered)



gaussian_filtered = cv2.GaussianBlur(gaussian_noisy,(5,5),0)    


cv2.imwrite("outputs/gaussian_filter.png", gaussian_filtered)


# task 4



def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    mse_value = mse(original, restored)
    if mse_value == 0:
        return 100
    return 20 * np.log10(255 / np.sqrt(mse_value))

# Gaussian Noise Evaluation
mse_mean = mse(gray, mean_filtered)
psnr_mean = psnr(gray, mean_filtered)

mse_gauss = mse(gray, gaussian_filtered)
psnr_gauss = psnr(gray, gaussian_filtered)

# Salt Pepper Evaluation
mse_median = mse(gray, median_filtered)
psnr_median = psnr(gray, median_filtered)

print("\n--- Gaussian Noise Restoration ---")
print(f"Mean Filter     -> MSE: {mse_mean:.2f}, PSNR: {psnr_mean:.2f}")
print(f"Gaussian Filter -> MSE: {mse_gauss:.2f}, PSNR: {psnr_gauss:.2f}")

print("\n--- Salt & Pepper Noise Restoration ---")
print(f"Median Filter   -> MSE: {mse_median:.2f}, PSNR: {psnr_median:.2f}")


# task 5 


if psnr_gauss > psnr_mean:
    print("Gaussian Filter performs better for Gaussian Noise.")
else:
    print("Mean Filter performs better for Gaussian Noise.")

print("Median Filter is best suited for Salt & Pepper Noise")
print("because it removes impulse noise while preserving edges.")



plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(gray, cmap='gray')
plt.title("Original")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(gaussian_noisy, cmap='gray')
plt.title("Gaussian Noise")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(sp_noisy, cmap='gray')
plt.title("Salt & Pepper Noise")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(mean_filtered, cmap='gray')
plt.title("Mean Filter")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(median_filtered, cmap='gray')
plt.title("Median Filter")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(gaussian_filtered, cmap='gray')
plt.title("Gaussian Filter")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/final_comparison.png")
plt.show()

print("\nAll outputs saved inside 'outputs/' folder.")
print("Project execution completed successfully!")
