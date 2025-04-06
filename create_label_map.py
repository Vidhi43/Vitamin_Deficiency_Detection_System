import os
import joblib

os.makedirs("model", exist_ok=True)

label_map = {
    0: "Vitamin A Deficiency",
    1: "Vitamin D Deficiency",
    2: "Vitamin B12 Deficiency"
}

joblib.dump(label_map, os.path.join("model", "label_map.pkl"))

print("✅ label_map.pkl created successfully!")