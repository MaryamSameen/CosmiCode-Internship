# Spam Email Classifier Web App

An AI-powered Spam Email Classifier built with Python, Scikit-learn, and Flask.  
Classifies emails as **Spam** or **Ham (Safe)** using a trained Multinomial Naive Bayes model with TF-IDF text vectorization.  
The app provides a clean, modern web UI to test live email text.

---

## Features

- Advanced text processing with TF-IDF vectorizer (NLP technique)
- Multinomial Naive Bayes for spam classification
- Simple, alluring web interface using Flask and CSS with glassy effects
- Real-time spam/ham prediction from user input text
- Save and load trained model with `joblib`

---

## Tech Stack

| Technology        | Purpose                         |
|-------------------|--------------------------------|
| Python            | Core language                  |
| Scikit-learn      | Machine Learning pipeline      |
| Flask             | Web server & UI                |
| Pandas            | Dataset handling               |
| Joblib            | Model saving/loading           |
| HTML + CSS        | User interface                 |

---

## Setup & Installation

1. **Clone the repository**
    ```
    git clone 
    cd spam_email_classifier
    ```

2. **Create a virtual environment and activate**
    ```
    python -m venv venv
    source venv/bin/activate      # Linux/Mac
    venv\Scripts\activate         # Windows
    ```

3. **Install dependencies**
    ```
    pip install -r requirements.txt
    ```

4. **Download dataset**
    - Download the SMS Spam Collection Dataset as `spam.csv` and place it in the project root.

5. **Train the model**
    ```
    python model.py
    ```
    - This saves the trained model as `spam_classifier.pkl`.

6. **Run the Flask app**
    ```
    python app.py
    ```

7. **Open browser**
    - Visit http://localhost:5000 to use the spam email classifier UI.

---

## File Overview

- `model.py`  
  Loads dataset, pre-processes using TF-IDF vectorizer, trains Naive Bayes model, and saves it.

- `app.py`  
  Flask backend to load model and serve web UI. Handles POST requests from users to classify email text.

- `templates/index.html`  
  User interface HTML with form for input and output result display.

- `static/style.css`  
  Modern styled CSS with glassmorphism for appealing UI.

- `spam.csv`  
  SMS Spam dataset (input data file).

- `requirements.txt`  
  Python dependencies list.

---



### Workflow

1. **Data Loading and Preprocessing**  
   - The `spam.csv` dataset contains labeled messages (`spam` or `ham`).
   - Text messages are vectorized into numerical features using `TfidfVectorizer` (NLP step).
   - Labels are encoded into binary format.

2. **Model Training**  
   - A Multinomial Naive Bayes model is trained on the TF-IDF vectors.
   - Dataset is split into training and test sets.
   - Model accuracy is printed.
   - The pipeline (vectorizer + model) is saved to disk.

3. **Prediction Service (Flask App)**  
   - On app start, the saved model pipeline is loaded.
   - Home page displays a form to enter email text.
   - User submits text; Flask routes POST to prediction logic.
   - The model predicts spam or ham.
   - Result displayed on the same page with stylish UI.

---



