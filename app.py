from flask import Flask,render_template, request
import numpy as np
import joblib as jb


app=Flask(__name__)

@app.route("/",methods=['GET','POST]'])

def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    TSH=(request.form['TSH'])
    FTI=(request.form['FTI'])
    TT4=(request.form['TT4'])
    T3=(request.form['T3'])
    sex=(request.form['sex'])
    pregnant=(request.form['pregnant'])
    psych=(request.form['psych'])
    goitre=(request.form['goitre'])

    
    
    arr=np.array([[TSH,FTI,TT4,T3,sex,pregnant,psych,goitre]])

    model=jb.load('RandomForestClassifier.pkl')     

    result=model.predict(arr)
    if result ==1:
        return render_template('after.html',data='you have thyroid')
    else:
      return render_template('after.html',data='you are normal')
    
    





if __name__ == '__main__':
    app.run()