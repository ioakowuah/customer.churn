from flask import Flask, render_template, request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


app = Flask(__name__)
#@app.route('/test')
#def test():
#    return "Flask is being used for development"

#Load model_prediction

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        try:
            Contract = float(request.form['Contract'])
            InternetService = float(request.form['InternetService'])
            PaymentMethod = float(request.form['PaymentMethod'])
            OnlineSecurity = float(request.form['OnlineSecurity'])
            TechSupport = float(request.form['TechSupport'])
            pred_args = [[Contract,InternetService,PaymentMethod,OnlineSecurity,TechSupport]]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1,-1)
            nav_base = open('nav_class_model.pkl','rb')
            ml_model = joblib.load(nav_base)
            model_prediction = ml_model.predict(pred_args_arr)
            model_prediction = round(float(model_prediction),0)
        except valueError:
            return "Please check if the values are entered correctly"
    return render_template('predict.html',prediction = model_prediction)

if __name__ =="__main__":
    app.run()
