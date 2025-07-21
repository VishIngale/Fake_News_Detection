# Fake_News_Detection

Fake News Detection Web App:
This is a simple and functional Fake News Detection Web Application built using Flask (Python) for the backend and HTML/CSS for the frontend. It uses Logistic Regression and TF-IDF vectorization to classify news articles as Real or Fake. Users can input text or upload .txt or .pdf files to detect fake news. It also includes basic user authentication with SQLite.

Features:
🧠 Machine Learning model (Logistic Regression)
📄 Supports input via text box, .txt, or .pdf files
📊 Displays prediction and confidence score
🔐 Login and Registration system (SQLite)
💾 Saves trained model and vectorizer using pickle
🧹 Preprocessing with TF-IDF vectorizer
🔐 Session-based authentication using Flask
🗂️ Simple folder structure and modular code

Tech Stack:
Frontend: HTML, CSS (via templates)
Backend: Flask, Python
Database: SQLite
Machine Learning: Scikit-learn (Logistic Regression, TF-IDF)
Libraries: pandas, numpy, PyPDF2

Sample Usage:
Login or register on the website,
Enter a news article or upload a .txt/.pdf file,
Click Predict,
View Real/Fake result and confidence score

Login Credentials:
Register a new account via the /register page,
Credentials are stored in users.db (SQLite)
