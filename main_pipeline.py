"""
End-to-End Industrial Quality Control Pipeline
1. Load image
2. Classify as normal/defective using KNN
3. If defective, suggest optimal decision using GAMS logic (simulated)
"""

import os
import random
from cv_module.defect_detector import detect_defect
from ml_module.knn_classifier import extract_features, load_dataset
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# Simulate a pre-trained KNN model (in real use, load from disk)
def load_pretrained_knn():
    X, y = load_dataset("dataset")
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    return knn

def simulate_gams_decision():
    """Simulate GAMS output: return best action and cost."""
    actions = ["Repair", "Recycle", "Discard"]
    costs = [round(random.uniform(5, 15), 2), 
             round(random.uniform(2, 8), 2), 
             round(random.uniform(20, 40), 2)]
    best_idx = np.argmin(costs)
    return actions[best_idx], costs[best_idx]

def run_pipeline(image_path):
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Step 1: Load and extract features
    feat = extract_features(image_path)
    if feat is None:
        print("❌ Error: Could not process image.")
        return

    # Step 2: Classify
    knn = load_pretrained_knn()
    pred = knn.predict([feat])[0]
    label = "Defective" if pred == 1 else "Normal"
    print(f"✅ Classification: {label}")

    # Step 3: If defective, run optimization logic
    if pred == 1:
        action, cost = simulate_gams_decision()
        print(f"💡 Optimal Decision: {action} (Estimated Cost: ${cost})")
    else:
        print("✅ Product is normal. No action needed.")

    # Optional: Show image (uncomment if running locally with GUI)
    # detect_defect(image_path)

if __name__ == "__main__":
    # Use a sample defective image
    sample_image = "dataset/defective/crack_001.jpg"
    if os.path.exists(sample_image):
        run_pipeline(sample_image)
    else:
        print("⚠️ Sample image not found. Please check dataset path.")