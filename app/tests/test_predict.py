from flask import Flask

from app.handlers.routes import configure_routes

# Testing general predict inputs
def test_predict_basic():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
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
    assert b"Invalid Parameters" in response.data

# Testing upper bounds of age variable
def test_predict_age_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "age": 23,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing lower bounds of health variable
def test_predict_health_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "health": 0,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing upper bounds of health variable
def test_predict_health_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "health": 6,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing lower bounds of absences variable
def test_predict_absences_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "absences": -1,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing upper bounds of absences variable
def test_predict_absences_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "absences": 94,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing lower bounds of studytime variable
def test_predict_studytime_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "studytime": 0,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing upper bounds of studytime variable
def test_predict_studytime_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "studytime": 5,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing lower bounds of failures variable
def test_predict_failures_OOB_low():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "failures": 0,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing upper bounds of failures variable
def test_predict_failures_OOB_high():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "failures": 5,
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing bounds of schoolsup variable
def test_predict_schoolsup_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "schoolsup": "maybe",
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing bounds of paid variable
def test_predict_paid_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "paid": "maybe",
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data

# Testing bounds of internet variable
def test_predict_internet_OOB():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()

    response = client.post("/predict", data={
        "internet": "maybe",
    })
    assert response.status_code == 400
    assert b"Invalid Parameters" in response.data
