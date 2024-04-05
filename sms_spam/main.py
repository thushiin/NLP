from flask import Flask,render_template,request
import joblib
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
m=open(r'/Users/thushinbhanu/Downloads/Maitexapython/nlp/joblib_model','rb')
mod=joblib.load(m)
c=open(r'/Users/thushinbhanu/Downloads/Maitexapython/nlp/countvect.joblib','rb')
cv=joblib.load(c)

label=['HAM','SPAM']

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('spam_sms.html')

@app.route('/predict',methods=['POST'])
def pred():
    if request.method=='POST':
        message=request.form['message']
        message=cv.transform([message])
        prediction = mod.predict(message)
        prediction=label[prediction[0]]
        return render_template("spam_sms.html", prediction=prediction)
    
if __name__ == "__main__":
    app.run(debug=True)