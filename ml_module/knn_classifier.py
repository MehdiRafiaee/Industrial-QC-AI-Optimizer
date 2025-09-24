"""
KNN Classifier for Industrial Defect Detection
- Extracts simple features (mean, std of pixel intensities)
- Trains a KNN model to classify images as 'normal' or 'defective'
- Uses dataset from /dataset/normal/ and /dataset/defective/
"""

import os
import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def extract_features(image_path):
    """Extract simple statistical features from image."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return [img.mean(), img.std()]

def load_dataset(dataset_path):
    """Load images and labels from dataset folders."""
    features = []
    labels = []
    
    # Load normal images (label = 0)
    normal_dir = os.path.join(dataset_path, "normal")
    for file in os.listdir(normal_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_features(os.path.join(normal_dir, file))
            if feat:
                features.append(feat)
                labels.append(0)  # 0 = normal

    # Load defective images (label = 1)
    defective_dir = os.path.join(dataset_path, "defective")
    for file in os.listdir(defective_dir):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            feat = extract_features(os.path.join(defective_dir, file))
            if feat:
                features.append(feat)
                labels.append(1)  # 1 = defective

    return np.array(features), np.array(labels)

def train_and_evaluate():
    # Load data
    X, y = load_dataset("dataset")
    print(f"Loaded {len(X)} samples.")

    if len(X) == 0:
        print("No images found! Check dataset paths.")
        return

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train KNN
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)

    # Predict
    y_pred = knn.predict(X_test)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    print(f"\n✅ KNN Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Defective"]))

    return knn

if __name__ == "__main__":
    model = train_and_evaluate()