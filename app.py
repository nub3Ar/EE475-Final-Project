from flask import Flask, render_template, request, redirect, url_for
from image_processor import tranformation
# from classifier.Main import CNN_prediction, CNN_Dropout_prediction, CNN_BatchNromalized_prediction

app = Flask(__name__)

fileName = "img1.jpg" 
choice = -1

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html', data=None)

@app.route('/chooseModel')
def chooseModel():
    return render_template('chooseModel.html')

@app.route('/error')
def error():
    return render_template('error.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
        fileName = uploaded_file.filename
        return redirect(url_for('chooseModel'))
    fileName = None 
    return redirect(url_for('index'))

@app.route('/chooseModel', methods=['POST'])
def stay_c():
    return redirect(url_for('chooseModel'))

@app.route('/result', methods=['POST'])
def stay_r():
    for key, value in request.form.items():
        choice = value
    matrix = tranformation(fileName)
    # if choice == 1:
    #     prediction = CNN_prediction(matrix)
    # elif choice == 2:
    #     prediction = CNN_Dropout_prediction(matrix)
    # else:
    #     prediction = CNN_BatchNromalized_prediction(matrix)
    return render_template('result.html', data=choice)

@app.route('/error', methods=['POST'])
def to_index():
    return redirect(url_for('index'))
