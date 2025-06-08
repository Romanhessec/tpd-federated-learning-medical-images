import cv2
import numpy as np
import matplotlib.pyplot as plt

# Path to your image (change this to any image you want to test)
image_path = "chexlocalize/CheXpert/test/patient64947/study1/view1_frontal.jpg"

# Read the image in grayscale
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError(f"Could not read image: {image_path}")

# Apply histogram equalization
img_eq = cv2.equalizeHist(img)

# Plot original and equalized images and their histograms
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

# Original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Histogram of original image
axs[1, 0].hist(img.ravel(), 256, range=(0, 256))
axs[1, 0].set_title('Histogram (Original)')

# Equalized image
axs[0, 1].imshow(img_eq, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

# Histogram of equalized image
axs[1, 1].hist(img_eq.ravel(), 256, range=(0, 256))
axs[1, 1].set_title('Histogram (Equalized)')

plt.tight_layout()
plt.show()

# Compute and sort pixel value counts for original image
orig_hist, orig_bins = np.histogram(img.ravel(), 256, [0, 256])
orig_sorted_idx = np.argsort(orig_hist)[::-1]
orig_sorted_counts = orig_hist[orig_sorted_idx]
orig_sorted_bins = orig_bins[:-1][orig_sorted_idx]

# Compute and sort pixel value counts for equalized image
hist_eq, bins_eq = np.histogram(img_eq.ravel(), 256, [0, 256])
eq_sorted_idx = np.argsort(hist_eq)[::-1]
eq_sorted_counts = hist_eq[eq_sorted_idx]
eq_sorted_bins = bins_eq[:-1][eq_sorted_idx]

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Original image
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

# Sorted histogram of original image
axs[1, 0].bar(range(256), orig_sorted_counts, color='gray')
axs[1, 0].set_title('Sorted Pixel Counts (Original)')
axs[1, 0].set_xlabel('Pixel Value (sorted by count)')
axs[1, 0].set_ylabel('Count')

# Equalized image
axs[0, 1].imshow(img_eq, cmap='gray')
axs[0, 1].set_title('Equalized Image')
axs[0, 1].axis('off')

# Sorted histogram of equalized image
axs[1, 1].bar(range(256), eq_sorted_counts, color='gray')
axs[1, 1].set_title('Sorted Pixel Counts (Equalized)')
axs[1, 1].set_xlabel('Pixel Value (sorted by count)')
axs[1, 1].set_ylabel('Count')

plt.tight_layout()
plt.show()

def histogram_entropy(hist):
    # Normalize to probabilities, avoid log(0) by filtering out zeros
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

orig_entropy = histogram_entropy(orig_hist)
eq_entropy = histogram_entropy(hist_eq)

print(f"Histogram Entropy (Original): {orig_entropy:.3f}")
print(f"Histogram Entropy (Equalized): {eq_entropy:.3f}")