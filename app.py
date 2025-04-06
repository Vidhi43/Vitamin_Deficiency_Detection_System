from flask import Flask, render_template, request, redirect, url_for, session
import sqlite3
import os
import bcrypt
from werkzeug.utils import secure_filename
import torch
from PIL import Image
import torchvision.transforms as transforms

# Initialize Flask App
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Set upload folder
UPLOAD_FOLDER = "static/uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Database setup
DB_PATH = "users.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT, email TEXT, phone TEXT, 
                            age INTEGER, password TEXT, gender TEXT)''')
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS image_analysis (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id INTEGER, image_path TEXT, 
                            prediction TEXT, analysis_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (user_id) REFERENCES users(id))''')
        conn.commit()

init_db()

# Load Model (Ensure vitamin_classifier.pth exists)
MODEL_PATH = "vitamin_classifier.pth"
if os.path.exists(MODEL_PATH):
    model = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
    model.eval()
else:
    model = None

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor()
])

def predict_image(image_path):
    if model is None:
        return "Model not loaded"
    
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output, 1)
    
    class_names = ["Vitamin A Deficiency", "Vitamin B Deficiency", "Vitamin C Deficiency"]  # Adjust based on training labels
    return class_names[predicted.item()]

# Flask Routes
@app.route("/")
def home():
    return render_template("index.html")  # Ensure this file is inside templates/

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        age = request.form["age"]
        password = bcrypt.hashpw(request.form["password"].encode(), bcrypt.gensalt())
        gender = request.form["gender"]
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, phone, age, password, gender) VALUES (?, ?, ?, ?, ?, ?)", 
                           (name, email, phone, age, password, gender))
            conn.commit()
        return redirect(url_for("login"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"].encode()
        
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, password FROM users WHERE email=?", (email,))
            user = cursor.fetchone()
            
            if user and bcrypt.checkpw(password, user[1]):
                session["user_id"] = user[0]
                return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if "user_id" not in session:
        return redirect(url_for("login"))

    if request.method == "POST":
        file = request.files["image"]
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(image_path)

            prediction = predict_image(image_path)

            with sqlite3.connect(DB_PATH) as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO image_analysis (user_id, image_path, prediction) VALUES (?, ?, ?)", 
                               (session["user_id"], image_path, prediction))
                conn.commit()
            
            return render_template("result.html", image=image_path, prediction=prediction)

    return render_template("upload.html")

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

if __name__ == "__main__":
    app.run(debug=True)



# ✅ Run Flask app with error handling
if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        print(f"Error: {e}")