from flask import Flask, render_template, request, redirect, url_for
from image_processor import tranformation
# from classifier.Main import CNN_prediction, CNN_Dropout_prediction, CNN_BatchNromalized_prediction

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
        matrix = tranformation(fileName)
    else: 
        return redirect(url_for('error'))
    # if choice == 1:
    #     prediction = CNN_prediction(matrix)s
    # elif choice == 2:
    #     prediction = CNN_Dropout_prediction(matrix)
    # else:
    #     prediction = CNN_BatchNromalized_prediction(matrix)
    return render_template('result.html', data=[fileName,choice])

@app.route('/error', methods=['POST'])
def to_index():
    return redirect(url_for('index'))
