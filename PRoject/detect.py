import cv2
import numpy as np
import pickle
from skimage.feature import local_binary_pattern, hog

# ── Config (must match train.py) ─────────────────────────
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

if __name__ == "__main__":
    # ── Load model & scaler ───────────────────────────────────
    try:
        with open("mask_classifier.pkl", "rb") as f:
            data = pickle.load(f)
            model = data['model']
            scaler = data['scaler']
    except FileNotFoundError:
        print("Model file not found. Run train.py first.")
        exit()

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # ── Webcam Loop ───────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open webcam.")
        exit()

    print("Press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.05,    # More sensitive
            minNeighbors=3,      # Detects at more angles
            minSize=(40, 40)     # Catches closer faces
        )


        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            features = extract_features(face_roi).reshape(1, -1)
            features = scaler.transform(features)   # Scale features

            pred = model.predict(features)[0]
            prob = model.predict_proba(features)[0][pred]

            label = "Mask" if pred == 1 else "No Mask"
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"{label} ({prob*100:.1f}%)",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, color, 2)

        cv2.imshow("Face Mask Compliance Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

