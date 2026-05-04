import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import pickle
from skimage.feature import local_binary_pattern, hog
from sklearn.metrics import confusion_matrix

# ── Configuration ───────────────────────────────────────
BASE_DIR = r"E:\Projects\facedetect"
ASSETS_DIR = os.path.join(BASE_DIR, "report_assets")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
IMG_WITH_PATH = os.path.join(BASE_DIR, r"dataset\with_mask\with_mask_2533.jpg")
IMG_WITHOUT_PATH = os.path.join(BASE_DIR, r"dataset\without_mask\without_mask_2473.jpg")

if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)

# ── 1. Create Dataset Distribution Bar Chart ──────────
print("Generating dataset_bar.jpg...")
# Stats provided in LaTeX source: 3725 vs 3828
categories = ['with_mask', 'without_mask']
counts = [3725, 3828]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, counts, color=['#2ecc71', '#e74c3c'])
plt.xlabel('Category', fontsize=12)
plt.ylabel('Number of Images', fontsize=12)
plt.title('Dataset Class Distribution', fontsize=14)

# Adding value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 50, yval, ha='center', va='bottom', fontsize=10)

plt.savefig(os.path.join(IMAGES_DIR, "dataset_bar.jpg"), dpi=150)
plt.close()

# ── 2. Handle Existing Assets (Rename/Copy) ──────────
# Mapping old names to LaTeX names
# appendix_b_s2_clahe.jpg -> clahe.jpg
# appendix_b_s3_lbp.jpg   -> lbp.jpg
# appendix_b_s4_hog.jpg   -> hog.jpg
# appendix_b_s5_confusion.jpg -> confusion.jpg

mapping = {
    "appendix_b_s2_clahe.jpg": "clahe.jpg",
    "appendix_b_s3_lbp.jpg": "lbp.jpg",
    "appendix_b_s4_hog.jpg": "hog.jpg",
    "appendix_b_s5_confusion.jpg": "confusion.jpg"
}

for old, new in mapping.items():
    src = os.path.join(ASSETS_DIR, old)
    dst = os.path.join(IMAGES_DIR, new)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"Copied {old} to {new}")
    else:
        print(f"Warning: {src} not found.")

# ── 3. Create Real-Time Detection Mockups (Mask/NoMask) 
print("Generating detections (mask.jpg, nomask.jpg)...")

def draw_detection(image_path, label, confidence, color):
    # Load image, resize for consistent framing
    img = cv2.imread(image_path)
    if img is None: return None
    img = cv2.resize(img, (600, 600))
    
    # Draw rectangle and text
    x, y, w, h = 100, 100, 400, 400 # Central mock ROI
    cv2.rectangle(img, (x, y), (x+w, y+h), color, 6)
    cv2.putText(img, f"{label} ({confidence}%)", (x, y - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    return img

# Green Detection (Mask 92.5%)
img_mask = draw_detection(IMG_WITH_PATH, "Mask", 92.5, (0, 255, 0))
if img_mask is not None:
    cv2.imwrite(os.path.join(IMAGES_DIR, "mask.jpg"), img_mask)

# Red Detection (No Mask 99.6%)
img_nomask = draw_detection(IMG_WITHOUT_PATH, "No Mask", 99.6, (0, 0, 255))
if img_nomask is not None:
    cv2.imwrite(os.path.join(IMAGES_DIR, "nomask.jpg"), img_nomask)

# ── 4. Flowchart Placeholder ──────────────────────────
# Creating a simple textual flowchart image if one doesn't exist
print("Generating placeholder flowchart.jpg...")
plt.figure(figsize=(10, 6), facecolor='white')
plt.axis('off')

# Simple Flow Boxes
boxes = [
    "Input Image\n(Webcam)", "Face Detection\n(Haar Cascades)",
    "Preprocessing\n(Resize + CLAHE)", "Feature Extraction\n(LBP + HOG)",
    "SVM Classification\n(Linear SVC)", "Output Display\n(Bounding Box)"
]

y_pos = 0.9
for i, box in enumerate(boxes):
    plt.text(0.5, y_pos, box, ha='center', va='center', fontsize=12,
             bbox=dict(boxstyle="round,pad=1.2", fc="#d6eaf8", ec="blue"))
    if i < len(boxes) - 1:
        plt.annotate("", xy=(0.5, y_pos - 0.1), xytext=(0.5, y_pos - 0.05),
                     arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    y_pos -= 0.15

plt.savefig(os.path.join(IMAGES_DIR, "flowchart.jpg"), dpi=150, bbox_inches='tight')
plt.close()

print("\nAll assets for LaTeX report are ready in the /images/ folder! 🚀")
