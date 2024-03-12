import pickle
import pandas as pd
from flask import Flask,render_template,request
application=Flask(__name__)
try:
    with open('loan.pkl','rb') as file:
        model=pickle.load(file)
except (IOError, EOFError) as e:
    print("Error loading the pickled file:", e)
@application.route('/')
def fun():
    return render_template('l3.html')
@application.route('/predict', methods=['POST'])
def predict():
    data=request.form
    features=[[float(data['Total_Income']),float(data['Credit_History']),
              float(data['LoanAmount'])]]
    prediction = model.predict(features)
    if prediction==1:
        output="eligible"
    elif prediction==0:
        output="not eligible"
    return render_template('l3.html',prediction_text="the person is {} for loan".format(output))
if __name__ == "__main__":
    application.run(debug=True)
