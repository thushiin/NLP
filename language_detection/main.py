from flask import Flask,render_template,request
import joblib

mod=joblib.load(r'/Users/thushinbhanu/Downloads/Maitexapython/nlp/language_detection/lang_detect.joblib')
tfidf=joblib.load(r'/Users/thushinbhanu/Downloads/Maitexapython/nlp/language_detection/tfidf.joblib')


classes={0:'Arabic',1:'Danish',2:'Dutch',3:'English',4:'French',5:'German',6:'Greek',7:'Hindi',8:'Italian',9:'Kannada',10:'Malayalam',11:'Portugeese',12:'Russian',13:'Spanish',14:'Sweedish',15:'Tamil',16:'Turkish'}

app=Flask(__name__)
@app.route('/')
def home():
    return render_template('language_detect.html')

@app.route('/predict',methods=['POST'])
def pred():
    if request.method=='POST':
        message=request.form['message']
        message=tfidf.transform([message])
        prediction=mod.predict(message)
        prediction=classes[prediction[0]]
        return render_template('language_detect.html',prediction=prediction)

if __name__=='__main__':
    app.run(debug=True)