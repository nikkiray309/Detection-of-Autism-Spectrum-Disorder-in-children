import traceback
from flask import Flask, render_template, request
from flask_cors import cross_origin
import pickle
import joblib
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET'])  
@cross_origin()
def homePage():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET']) 
@cross_origin()
def predict():
    if request.method == 'POST':
        try:
            # reading the inputs given by the user
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            eth = request.form['eth']
            jaun = int(request.form['q0'])
            q1 = int(request.form['q1'])
            q2 = int(request.form['q2'])
            q3 = int(request.form['q3'])
            q4 = int(request.form['q4'])
            q5 = int(request.form['q5'])
            q6 = int(request.form['q6'])
            q7 = int(request.form['q7'])
            q8 = int(request.form['q8'])
            q9 = int(request.form['q9'])
            q10 = int(request.form['q10'])
            fm = int(request.form['q11'])
            text2 = request.form['text2']
            #print(age,sex,eth,jaun,q1, q2, q3, q4, q5, q6, q7, q8, q9,q10,fm,text2)

            features = [q1, q2, q3, q4, q5, q6, q7, q8, q9,q10,age,sex,jaun,fm]
            #print(type(q2))
            #Q_chat_Score=sum(asd1)
            

            ethi_freq = {
                'Hispanic': 0,
                'Latino': 0,
                'Native Indian': 0,
                'Others': 0,
                'Pacifica': 0,
                'White European': 0,
                'asian': 0,
                'black': 0,
                'middle eastern': 0,
                'mixed': 0,
                'south asian': 0
            }
            ethi_freq[eth] = 1
            ethin_list = []
            for val in ethi_freq.values():
                features.append(val)
                ethin_list.append(val)
            
            print(features)
            print(ethin_list)
            model = joblib.load('random_forest_model.joblib')
            print('MODEL HAS BEEN LOADED!!-------------------------------------------------')
            prediction = model.predict([features])

            if prediction == 1:
                pred = "Your child may have Autism Spectrum Disorder. We recommend consulting a specialist."
            else:
                pred = "Your child appears to be typically developing."

            # Render the result template with prediction'''
            return render_template('results.html', prediction=pred)
        except Exception as e:
            print('The Exception message is: ', e)
            traceback.print_exc()
            return 'Something went wrong while processing the prediction.'
    else:
        return render_template('index.html')

     

if __name__ == "__main__":
    app.run(debug=True)