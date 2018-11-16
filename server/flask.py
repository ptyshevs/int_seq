from flask import Flask, jsonify, render_template, request
import numpy as np
import sys
sys.path.append('../..')

app = Flask(__name__)

@app.route('/')
def entry_point():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def form_process():
    text = request.form['text']
    # print(text)
    prepare_sequence(text)
    return render_template('index.html', form=form_process, text=text)

def prepare_sequence(text):
    values = list(map(int, text.split(',')))
    print(values)


@app.route('/ping', methods=['GET'])
def another():
    return jsonify('hi')

@app.route('/page')
def static_page():
    return render_template('index.html')
