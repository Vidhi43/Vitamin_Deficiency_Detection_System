import os
import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

dataset_path = "dataset"
X = []
y = []
label_map = {}
label_index = 0

print("🔍 Scanning dataset...")

# Loop through folders
for class_folder in sorted(os.listdir(dataset_path)):
    class_path = os.path.join(dataset_path, class_folder)
    if os.path.isdir(class_path):
        print(f"📁 Found class: {class_folder}")
        label_map[label_index] = class_folder
        image_count = 0
        for img_file in os.listdir(class_path):
            try:
                img_path = os.path.join(class_path, img_file)
                img = Image.open(img_path).convert("RGB").resize((64, 64))
                features = np.array(img).mean(axis=(0, 1)) / 255.0  # RGB mean
                X.append(features)
                y.append(label_index)
                image_count += 1
            except Exception as e:
                print(f"⚠️ Error reading {img_file}: {e}")
        print(f"🖼️ Processed {image_count} images for class '{class_folder}'")
        label_index += 1

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

if len(X) == 0:
    print("❌ No data found! Please check your dataset folder.")
    exit()

# Train and save model
model = RandomForestClassifier()
model.fit(X, y)

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/vitamin_model.pkl")
joblib.dump(label_map, "model/label_map.pkl")

print("✅ Model and label map saved!")
