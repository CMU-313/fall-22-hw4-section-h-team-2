from flask import Flask

from app.handlers.routes import configure_routes


def test_base_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/'

    response = client.get(url)

    assert response.status_code == 200
    assert response.get_data() == b'try the predict route it is great!'

def test_wipe_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/wipe'

    response = client.get(url)

    assert response.status_code == 200
    assert b'Model wiped' in response.data

def test_train_route():
    app = Flask(__name__)
    configure_routes(app)
    client = app.test_client()
    url = '/train'

    response = client.get(url)

    assert response.status_code == 200
    assert b'Model training score:' in response.data
