from flask import Flask, request, render_template
from wtforms import Form, FloatField, validators
import requests
import joblib
from model import process_input

app = Flask(__name__)
# Model

#view
@app.route('/',methods=['GET','POST'])
def question():
    if request.method == 'POST':
        inputname = str(request.form.get("name"))
        result, female, male = predict(inputname)
    else:
        inputname = ''
        result, female, male = '','',''
    return render_template("view.html", inputname=inputname, female=female, male=male, result=result)

#compute
def predict(s):
       if lr:
         try:
            name_input = process_input(s)
            result = lr.predict(name_input)
            result_proba = lr.predict_proba(name_input)
            return result, round(result_proba[0][0],3), round(result_proba[0][1],3)
         except:
            return 'none','none','none'


if __name__ == '__main__':
    lr = joblib.load("model.pkl") # Load "model.pkl"
    print ('Model loaded')
    app.run()