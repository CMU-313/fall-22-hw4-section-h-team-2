from flask import Flask
import json

from app.handlers.routes import configure_routes

# Testing general predict inputs
def test_predict_basic():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", json={
        "age": 18,
        "health": 4,
        "absences": 3,
        "studytime": 2,
        "failures": 0,
        "schoolsup": "no",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 200
    assert b"Successful operation" in response.data

# Testing lower bounds of age variable
def test_predict_age_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 14,
        "health": 3,
        "absences": 7,
        "studytime": 4,
        "failures": 0,
        "schoolsup": "yes",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing upper bounds of age variable
def test_predict_age_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 23,
        "health": 5,
        "absences": 2,
        "studytime": 4,
        "failures": 0,
        "schoolsup": "no",
        "paid": "yes",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing lower bounds of health variable
def test_predict_health_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 20,
        "health": 0,
        "absences": 21,
        "studytime": 3,
        "failures": 1,
        "schoolsup": "yes",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing upper bounds of health variable
def test_predict_health_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 19,
        "health": 6,
        "absences": 21,
        "studytime": 2,
        "failures": 2,
        "schoolsup": "no",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing lower bounds of absences variable
def test_predict_absences_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 18,
        "health": 4,
        "absences": -1,
        "studytime": 4,
        "failures": 0,
        "schoolsup": "no",
        "paid": "yes",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing upper bounds of absences variable
def test_predict_absences_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 17,
        "health": 2,
        "absences": 94,
        "studytime": 4,
        "failures": 1,
        "schoolsup": "no",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing lower bounds of studytime variable
def test_predict_studytime_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 17,
        "health": 4,
        "absences": 12,
        "studytime": 0,
        "failures": 3,
        "schoolsup": "no",
        "paid": "no",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing upper bounds of studytime variable
def test_predict_studytime_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 21,
        "health": 4,
        "absences": 0,
        "studytime": 5,
        "failures": 0,
        "schoolsup": "no",
        "paid": "yes",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing lower bounds of failures variable
def test_predict_failures_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 22,
        "health": 3,
        "absences": 5,
        "studytime": 4,
        "failures": -1,
        "schoolsup": "yes",
        "paid": "yes",
        "internet": "no",
    })
    assert response.status_code == 400

# Testing upper bounds of failures variable
def test_predict_failures_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 20,
        "health": 5,
        "absences": 54,
        "studytime": 1,
        "failures": 5,
        "schoolsup": "yes",
        "paid": "no",
        "internet": "no",
    })
    assert response.status_code == 400

# Testing bounds of schoolsup variable
def test_predict_schoolsup_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 16,
        "health": 3,
        "absences": 9,
        "studytime": 2,
        "failures": 1,
        "schoolsup": "maybe",
        "paid": "yes",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing bounds of paid variable
def test_predict_paid_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 18,
        "health": 5,
        "absences": 2,
        "studytime": 4,
        "failures": 0,
        "schoolsup": "yes",
        "paid": "maybe",
        "internet": "yes",
    })
    assert response.status_code == 400

# Testing bounds of internet variable
def test_predict_internet_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 20,
        "health": 4,
        "absences": 5,
        "studytime": 3,
        "failures": 0,
        "schoolsup": "no",
        "paid": "no",
        "internet": "maybe",
    })
    assert response.status_code == 400
