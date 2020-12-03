from flask import Flask, render_template, request, redirect, url_for
from classifier.models import try_connect

app = Flask(__name__)

@app.route('/')
def index():
    print(try_connect())
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/', methods=['POST'])
def upload_file():
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        uploaded_file.save(uploaded_file.filename)
    return redirect(url_for('result'))

@app.route('/result', methods=['POST'])
def go_back():
    return redirect(url_for('index'))
