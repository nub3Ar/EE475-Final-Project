import os
print(os.getcwd())

from flask import Flask, render_template, request, redirect, url_for
from image_processor import transformation
from classifier.Main import CNN_prediction, CNN_Dropout_prediction, CNN_BatchNromalized_prediction
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html', data=None)

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/', methods=['POST'])
def upload_file():
    return redirect(url_for('index'))

@app.route('/result', methods=['POST'])
def predict():
    fileName, choice = None, None
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        fileName = uploaded_file.filename
    for key, value in request.form.items():
        choice = value
    if fileName:
        matrix = transformation(fileName)
        matrix = np.array(matrix)
        matrix = matrix.astype("float32") / 255
        matrix = np.expand_dims(matrix, -1)
    else: 
        return redirect(url_for('error'))
    print(type(choice))
    if choice == "CNN":
        probability, prediction = CNN_prediction(matrix)
        print('1')
    elif choice == "CNN_Dropout":
        print('2')
        probability, prediction = CNN_Dropout_prediction(matrix)
    else:
        print('3')
        probability, prediction = CNN_BatchNromalized_prediction(matrix)
    
    prediction = ''.join([str(x) for x in prediction])


    return render_template('result.html', data=[prediction, probability])

@app.route('/error', methods=['POST'])
def to_index():
    return redirect(url_for('index'))
