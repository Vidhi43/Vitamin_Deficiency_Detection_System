import joblib

label_map = {
    0: "Vitamin A Deficiency",
    1: "Vitamin B Deficiency",
    2: "Vitamin C Deficiency"
}

joblib.dump(label_map, "model/label_map.pkl")
