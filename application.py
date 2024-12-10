
from flask import Flask,request,jsonify,render_template
import pickle
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

## import ridge and scaler pickle

ridge_model = pickle.load(open('/Users/prabin/python_study/machine_learning_projects/models/ridge.pkl','rb'))
scaler_model = pickle.load(open('/Users/prabin/python_study/machine_learning_projects/models/scaler.pkl','rb'))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoints():
    if request.method == 'POST':
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))  # Assuming this is a string input
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        Ws = float(request.form.get('Ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        Region = float(request.form.get('Region'))  # Assuming this is a string input
        
        new_data_scaled = scaler_model.transform([[DMC,ISI,Classes,Temperature,RH,Ws,Rain,FFMC,Region]])
        result = ridge_model.predict(new_data_scaled)
        return render_template('home.html',results = result[0])
    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=8000)