from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('spam_classifier.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        email_text = request.form['email_text']
        pred = model.predict([email_text])[0]
        prediction = "Spam" if pred == 1 else "Ham (Not Spam)"
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
