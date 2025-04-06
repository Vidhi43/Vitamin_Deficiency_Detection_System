import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

dataset_path = "dataset"
X, y = [], []
label_map = {}
label_index = 0

# Sort ensures labels are always in same order
for folder in sorted(os.listdir(dataset_path)):
    folder_path = os.path.join(dataset_path, folder)
    if os.path.isdir(folder_path):
        label_map[label_index] = folder  # 0: "vitamin A", etc.
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            try:
                img = Image.open(img_path).convert("RGB").resize((64, 64))
                features = np.array(img) / 255.0
                X.append(features.mean(axis=(0, 1)))  # RGB avg
                y.append(label_index)
            except:
                pass
        label_index += 1

X, y = np.array(X), np.array(y)

# Train and save
model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/vitamin_model.pkl")
joblib.dump(label_map, "model/label_map.pkl")

print("✅ Model trained and saved!")
print("Label map:", label_map)

