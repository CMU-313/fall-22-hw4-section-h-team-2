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
        include = ['health', 'absences','age', 'studytime', 'failures', 'paid', 'schoolsup', 'internet', 'G3']
        df.drop(columns=df.columns.difference(include), inplace=True)  # only using above features
        df['qual_student'] = np.where(df['G3']>=15, 1, 0)
        include = ['health', 'absences','age', 'studytime', 'failures', 'paid', 'schoolsup', 'internet','qual_student']
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
        age = request.args.get('age')
        absences = request.args.get('absences')
        health = request.args.get('health')
        studytime = request.args.get('studytime')
        failures = request.args.get('failures')
        schoolsup = request.args.get('schoolsup')
        paid = request.args.get('paid')
        internet = request.args.get('internet')

        schoolsup_nan = 0
        schoolsup_yes = 0
        schoolsup_no = 0

        if (schoolsup == 'no'):
            schoolsup_no = 1
        elif (schoolsup == 'yes'):
            schoolsup_yes = 1
        else:
            schoolsup_nan = 1
        
        paid_nan = 0
        paid_yes = 0
        paid_no = 0

        if (paid == 'no'):
            paid_no = 1
        elif (paid == 'yes'):
            paid_yes = 1
        else:
            paid_nan = 1
        
        internet_nan = 0
        internet_yes = 0
        internet_no = 0

        if (internet == 'no'):
            internet_no = 1
        elif (internet == 'yes'):
            internet_yes = 1
        else:
            internet_nan = 1

        data = [[age], [health], [absences], [studytime], [failures], [schoolsup_nan], [schoolsup_yes], [schoolsup_no], [paid_nan], [paid_yes], [paid_no], [internet_nan], [internet_yes], [internet_no]]
        query_df = pd.DataFrame({
            'age': pd.Series(age),
            'health': pd.Series(health),
            'absences': pd.Series(absences),
            'studytime': pd.Series(studytime),
            'failures': pd.Series(failures),
            'schoolsup_nan': pd.Series(schoolsup_nan),
            'schoolsup_yes': pd.Series(schoolsup_yes),
            'schoolsup_no': pd.Series(schoolsup_no),
            'paid_nan': pd.Series(paid_nan),
            'paid_yes': pd.Series(paid_yes),
            'paid_no': pd.Series(paid_no),
            'internet_nan': pd.Series(internet_nan),
            'internet_yes': pd.Series(internet_yes),
            'internet_no': pd.Series(internet_no)
        })
        query = pd.get_dummies(query_df)
        prediction = clf.predict(query)
        return jsonify(np.asscalar(prediction))
        '''
        json_ = request.json
        query_df = pd.DataFrame(json_)
        query = pd.get_dummies(query_df)
        prediction = clf.predict(query)
        return jsonify({'prediction': list(prediction)})
        '''
    
    @app.route('/wipe', methods=['GET'])
    def wipe():
        try:
            shutil.rmtree('model')
            os.makedirs(this_dir)
            return 'Model wiped'

        except Exception as e:
            print(str(e))
            return 'Could not remove and recreate the model directory'
