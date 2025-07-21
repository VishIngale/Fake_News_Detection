from flask import Flask, render_template, request, redirect, url_for, session, flash
import pickle
import numpy as np
import sqlite3
from werkzeug.utils import secure_filename
import PyPDF2
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# Load ML model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf'}

# Initialize SQLite database
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT NOT NULL UNIQUE,
        password TEXT NOT NULL
    )''')
    conn.commit()
    conn.close()

init_db()

# Check allowed file type
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if 'username' not in session:
        flash("You must be logged in to use this feature.")
        return redirect(url_for('login'))

    prediction = None
    confidence = None
    text = ""

    if request.method == 'POST':
        file = request.files.get('file')
        if file and allowed_file(file.filename):
            ext = os.path.splitext(file.filename)[1].lower()
            if ext == '.txt':
                text = file.read().decode("utf-8")
            elif ext == '.pdf':
                pdf_reader = PyPDF2.PdfReader(file)
                text = " ".join([page.extract_text() or "" for page in pdf_reader.pages])
        else:
            text = request.form.get('news', '').strip()

        if text:
            vect = vectorizer.transform([text])
            proba = model.predict_proba(vect)[0]  # [Fake%, Real%]
            label = model.predict(vect)[0]        # 0 = Fake, 1 = Real
            confidence = round(np.max(proba) * 100, 2)
            prediction = "Real News ✅" if label == 1 else "Fake News ❌"

    return render_template('predict.html', prediction=prediction, confidence=confidence, text=text)

# Login
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session['username'] = username
            flash("Login successful!")
            return redirect(url_for('predict'))
        else:
            flash("Invalid credentials. Please try again.")
            return redirect(url_for('login'))

    return render_template("login.html")

# Register
@app.route('/register', methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            flash("Registration successful! You can now login.")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username already exists. Try another.")
            return redirect(url_for('register'))
        finally:
            conn.close()
    return render_template("register.html")

# Logout
@app.route('/logout')
def logout():
    session.pop('username', None)
    flash("You have been logged out.")
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

