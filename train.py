import os
import numpy as np
import cv2
from skimage.feature import local_binary_pattern, hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import pickle

# ── Better LBP Config ─────────────────────────────────────
RADIUS = 3
N_POINTS = 8 * RADIUS
IMG_SIZE = (64, 64)

def extract_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    # LBP features
    lbp = local_binary_pattern(gray, N_POINTS, RADIUS, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, N_POINTS + 3),
                           range=(0, N_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)

    # HOG features
    hog_feat = hog(gray, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)

    # Combine both
    return np.concatenate([hist, hog_feat])

def load_dataset(data_dir):
    X, y = [], []
    categories = {'with_mask': 1, 'without_mask': 0}
    for label, cls in categories.items():
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"Directory {folder} does not exist.")
            continue
        for fname in os.listdir(folder):
            img_path = os.path.join(folder, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            features = extract_features(img)
            X.append(features)
            y.append(cls)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # ── Load & Train ──────────────────────────────────────────
    print("Loading dataset...")
    X, y = load_dataset("dataset")

    if len(X) == 0:
        print("Dataset is empty. Please add images to dataset/with_mask and dataset/without_mask.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)

        print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

        # Scale features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        print("Training SVM (may take 2-3 mins)...")
        svm = SVC(kernel='rbf', C=100, gamma='scale', probability=True)
        svm.fit(X_train, y_train)

        # ── Evaluate ──────────────────────────────────────────────
        y_pred = svm.predict(X_test)
        print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
        print(classification_report(y_test, y_pred,
              target_names=['without_mask', 'with_mask']))

        # ── Save Model + Scaler ───────────────────────────────────
        with open("mask_classifier.pkl", "wb") as f:
            pickle.dump({'model': svm, 'scaler': scaler}, f)
        print("Model saved to mask_classifier.pkl")

