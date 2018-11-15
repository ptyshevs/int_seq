from flask import Flask, jsonify, render_template, request, flash

app = Flask(__name__)

@app.route('/')
def entry_point():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def form_process():
    text = request.form['text']
    # print(text)
    print(request.values)
    return render_template('index.html', form=form_process, text=text)



@app.route('/ping', methods=['GET'])
def another():
    return jsonify('hi')

@app.route('/page')
def static_page():
    return render_template('index.html')
