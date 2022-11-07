import this
from flask import Flask, jsonify, request
import joblib
import pandas as pd
import numpy as np
import os
import shutil
from sklearn.ensemble import RandomForestClassifier as rf
import sklearn
import time
import traceback
import json

# These will be populated at training time
model_columns = None
clf = None

def configure_routes(app):

    this_dir = os.path.dirname(__file__)
    model_path = os.path.join(this_dir, "model.pkl")
    clf = joblib.load(model_path)

    @app.route('/')
    def hello():
        return "try the predict route it is great!"

    @app.route('/train', methods=['GET'])
    def train():
        df = pd.read_csv('data/student-mat.csv', sep=';')
        include = ['age', 'health', 'absences', 'studytime', 'failures', 'paid', 'schoolsup', 'internet', 'G3']
        df.drop(columns=df.columns.difference(include), inplace=True)  # only using above features
        df['qual_student'] = np.where(df['G3']>=15, 1, 0)
        include = ['age', 'health', 'absences', 'studytime', 'failures', 'paid', 'schoolsup', 'internet','qual_student']
        df.drop(columns=df.columns.difference(include), inplace=True)

        binaries = []
        for col, col_type in df.dtypes.iteritems():
            if col_type == 'O':
                binaries.append(col)
        
        df_ohe = pd.get_dummies(df, columns=binaries, dummy_na=True)

        dependent_variable = 'qual_student'
        x = df_ohe[df_ohe.columns.difference([dependent_variable])]
        y = df_ohe[dependent_variable]

        clf = rf(n_estimators = 1000)
        start = time.time()
        clf.fit(x, y)

        joblib.dump(clf, 'app/handlers/model.pkl')

        message1 = 'Trained in %.5f seconds' % (time.time() - start)
        message2 = 'Model training score: %s' % clf.score(x, y)
        return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
        return return_message


    @app.route('/predict', methods=['POST'])
    def predict():
        #use entries from the query string here but could also use json
        student_info = request.json

        if (student_info['age'] < 15 or student_info['age'] > 22):
            return "Invalid parameters", 400
        
        if (student_info['health'] < 1 or student_info['health'] > 5):
            return "Invalid parameters", 400
        
        if (student_info['absences'] < 0 or student_info['absences'] > 93):
            return "Invalid parameters", 400
        
        if (student_info['studytime'] < 1 or student_info['studytime'] > 4):
            return "Invalid parameters", 400
        
        if (student_info['failures'] < 0 or student_info['failures'] > 4):
            return "Invalid parameters", 400
        
        if (student_info['schoolsup'] == 'no'):
            student_info.update({'schoolsup_nan': 0})
            student_info.update({'schoolsup_yes': 0})
            student_info.update({'schoolsup_no': 1})
        elif (student_info['schoolsup'] == 'yes'):
            student_info.update({'schoolsup_nan': 0})
            student_info.update({'schoolsup_yes': 1})
            student_info.update({'schoolsup_no': 0})
        elif (student_info['schoolsup'] == None):
            student_info.update({'schoolsup_nan': 1})
            student_info.update({'schoolsup_yes': 0})
            student_info.update({'schoolsup_no': 0})
        else:
            return "Invalid parameters", 400
        del student_info['schoolsup']
        
        if (student_info['paid'] == 'no'):
            student_info.update({'paid_nan': 0})
            student_info.update({'paid_yes': 0})
            student_info.update({'paid_no': 1})
        elif (student_info['paid'] == 'yes'):
            student_info.update({'paid_nan': 0})
            student_info.update({'paid_yes': 1})
            student_info.update({'paid_no': 0})
        else:
            student_info.update({'paid_nan': 1})
            student_info.update({'paid_yes': 0})
            student_info.update({'paid_no': 0})
        del student_info['paid']
        
        if (student_info['internet'] == 'no'):
            student_info.update({'internet_nan': 0})
            student_info.update({'internet_yes': 0})
            student_info.update({'internet_no': 1})
        elif (student_info['internet'] == 'yes'):
            student_info.update({'internet_nan': 0})
            student_info.update({'internet_yes': 1})
            student_info.update({'internet_no': 0})
        else:
            student_info.update({'internet_nan': 1})
            student_info.update({'internet_yes': 0})
            student_info.update({'internet_no': 0})
        del student_info['internet']

        query_df = pd.DataFrame(student_info, index=[0])
        query = pd.get_dummies(query_df)
        prediction = clf.predict(query)

        return "Successful operation. Prediction: %s" % jsonify(np.ndarray.item(prediction))
