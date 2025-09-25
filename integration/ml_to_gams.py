"""
Bridge between ML output and GAMS optimization model.
- Simulates ML prediction (0=normal, 1=defective)
- Generates GAMS data file (.gdx or .dat) for defective items
"""

import numpy as np
import os

def generate_gams_data(ml_predictions, output_path="optimization_module/defect_data.inc"):
    """
    Generate GAMS include file from ML predictions.
    In real use, this would use actual defect probabilities and costs.
    """
    defective_indices = [i for i, pred in enumerate(ml_predictions) if pred == 1]
    
    if not defective_indices:
        print("No defective items detected. Optimization not needed.")
        return

    with open(output_path, 'w') as f:
        f.write("* Auto-generated from ML predictions\n")
        f.write("defect_prob(i) = (\n")
        for i in defective_indices:
            prob = np.random.uniform(0.6, 0.95)  # Simulated high confidence
            f.write(f"    p{i+1} {prob:.2f}\n")
        f.write(");\n")
    
    print(f"✅ GAMS data file generated: {output_path}")

# Simulate ML output (e.g., from KNN)
if __name__ == "__main__":
    # Example: 10 products, 4 are defective
    simulated_ml_output = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
    generate_gams_data(simulated_ml_output)