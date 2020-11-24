import flask
from flask import Flask, request, render_template, Response
from flask_cors import CORS

from generator_distilgpt2 import service

app = flask.Flask(__name__)
CORS(app)


@app.route("/predict", methods=['POST'])
def predict():
    data = {"success": False}
    input_json = request.json
    try:
        data_provider = service(input_json)
        data = data_provider
        data['success'] = True
        return data
    except Exception as exc:
        data["prediction_Exception"] = str(exc)
        return data
    

# if this is the main thread of execution first load the model and
@app.route("/")
def homepage():
    return "Welcome to Test Data Generator!"


if __name__ == "__main__":
    app.run()