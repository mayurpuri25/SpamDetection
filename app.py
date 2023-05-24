from flask import Flask, render_template, request
from joblib import load
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  # provide stemming to a words Eg. (Loved,Loving) -> love
from tensorflow.python.keras.models import load_model

app=Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

def spam(msg):
    cv = load('./static/files/cv.joblib')
    model = load_model('./static/files/trained_model.h5', compile=False)
    corpus_new = []
    review = msg.split()
    stop_words = set(stopwords.words('english'))
    pa = PorterStemmer()
    review = [pa.stem(word) for word in review if word not in stop_words]
    review = " ".join(review)
    corpus_new.append(review)
    new_x = cv.transform(corpus_new).toarray()
    ans = (model.predict(new_x) > 0.5)
    if ans:
        return 1
    else:
        return 0

@app.route('/', methods=["POST"])
def predict():
    if request.method == 'POST':
        text = request.form.get("sms")

        if text == '':
            return render_template('index.html')
        else:
            res=spam(text)
            if res:
                pred = 'SPAM'
            else:
                pred = "NOT SPAM"
            return render_template('index.html', result= pred, sms= text)
        
    else:
        return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)