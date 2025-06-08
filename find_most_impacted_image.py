import cv2
import numpy as np
import os

def histogram_entropy(hist):
    prob = hist / np.sum(hist)
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob))

def find_most_impacted_image(root_dir):
    max_diff = -1
    max_file = None
    max_orig_entropy = None
    max_eq_entropy = None
    total_files = sum(len([f for f in files if f.lower().endswith('.jpg')]) for _, _, files in os.walk(root_dir))
    processed = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.lower().endswith('.jpg'):
                processed += 1
                print(f"Processing {processed}/{total_files}: {os.path.join(dirpath, fname)}", end='\r')
                fpath = os.path.join(dirpath, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                orig_hist, _ = np.histogram(img.ravel(), 256, [0, 256])
                orig_entropy = histogram_entropy(orig_hist)
                img_eq = cv2.equalizeHist(img)
                eq_hist, _ = np.histogram(img_eq.ravel(), 256, [0, 256])
                eq_entropy = histogram_entropy(eq_hist)
                diff = abs(eq_entropy - orig_entropy)
                if diff > max_diff:
                    max_diff = diff
                    max_file = fpath
                    max_orig_entropy = orig_entropy
                    max_eq_entropy = eq_entropy
    print()  # for newline after progress
    return max_file, max_orig_entropy, max_eq_entropy, max_diff

if __name__ == "__main__":
    root_dir = "chexlocalize/CheXpert/test/"
    result = find_most_impacted_image(root_dir)
    if result[0] is not None:
        print(f"Image with most impact: {result[0]}")
        print(f"Original entropy: {result[1]:.3f}")
        print(f"Equalized entropy: {result[2]:.3f}")
        print(f"Absolute entropy difference: {result[3]:.3f}")
    else:
        print("No images found or all failed to process.")
