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


    @app.route('/predict')
    def predict():
        '''
        if clf:
            try:
                json_ = request.json
                query = pd.get_dummies(pd.DataFrame(json_))

                # https://github.com/amirziai/sklearnflask/issues/3
                # Thanks to @lorenzori
                query = query.reindex(columns=model_columns, fill_value=0)

                prediction = list(clf.predict(query))

                # Converting to int from int64
                return jsonify({"prediction": list(map(int, prediction))})

            except Exception as e:

                return jsonify({'error': str(e), 'trace': traceback.format_exc()})
        else:
            print('train first')
            return 'no model here'
        '''
        #use entries from the query string here but could also use json
        age = request.args.get('age')
        absences = request.args.get('absences')
        health = request.args.get('health')
        studytime = request.args.get('studytime')
        failures = request.args.get('failures')
        schoolsup = request.args.get('schoolsup')
        paid = request.args.get('paid')
        internet = request.args.get('internet')
        data = [[age], [health], [absences], [studytime], [failures], [schoolsup], [paid], [internet]] #[schoolsup_nan], [schoolsup_yes], [schoolsup_no], [paid_nan], [paid_yes], [paid_no], [internet_nan], [internet_yes], [internet_no]
        query_df = pd.DataFrame({
            'age': pd.Series(age),
            'health': pd.Series(health),
            'absences': pd.Series(absences),
            'studytime': pd.Series(studytime),
            'failures': pd.Series(failures),
            'schoolsup': pd.Series(schoolsup),
            'paid': pd.Series(paid),
            'internet': pd.Series(internet)
            #'schoolsup_nan': pd.Series(schoolsup_nan),
            #'schoolsup_yes': pd.Series(schoolsup_yes),
            #'schoolsup_no': pd.Series(schoolsup_no),
            #'paid_nan': pd.Series(paid_nan),
            #'paid_yes': pd.Series(paid_yes),
            #'paid_no': pd.Series(paid_no),
            #'internet_nan': pd.Series(internet_nan),
            #'internet_yes': pd.Series(internet_yes),
            #'internet_no': pd.Series(internet_no)
        })
        query = pd.get_dummies(query_df)
        prediction = clf.predict(query)
        return jsonify(np.asscalar(prediction))
    
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

        global model_columns
        model_columns = list(x.columns)
        joblib.dump(clf, 'app/handlers/model.pkl')

        global clf
        clf = rf(n_estimators = 1000)
        start = time.time()
        clf.fit(x, y)

        message1 = 'Trained in %.5f seconds' % (time.time() - start)
        message2 = 'Model training score: %s' % clf.score(x, y)
        return_message = 'Success. \n{0}. \n{1}.'.format(message1, message2) 
        return return_message
    
    @app.route('/wipe', methods=['GET'])
    def wipe():
        try:
            shutil.rmtree('model')
            os.makedirs('model')
            return 'Model wiped'

        except Exception as e:
            print(str(e))
            return 'Could not remove and recreate the model directory'
