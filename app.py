from flask import Flask, request, render_template
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

@app.route("/", methods=['GET'])
def home():
    return render_template("home.html")  # You can create a simple home page if needed

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    # Load dataset
    df = pd.read_csv('spam.csv', encoding='latin-1')
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.columns = ['label', 'text']  # Rename columns based on the dataset you shared

    # Map labels to numerical values
    df['label'] = df['label'].map({"ham": 0, "spam": 1})

    # Features and labels
    X = df['text']
    y = df['label']

    # Text vectorization
    cv = CountVectorizer()
    X = cv.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes classifier
    clf = MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)

    # Predict based on user input
    if request.method == "POST":
        message = request.form['message']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)

        return render_template('predict.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
