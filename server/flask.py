from flask import Flask, jsonify, render_template, request
import sys
sys.path.append('../..')
from models.pipeline import Pipeline
from models.baseline import Baseline
from models.diff_table import DiffTable
from models.linear_model import LinearModel
from models.Nonlinear_model import NonLinearModel
from models.markov_chain import MarkovChain
from models.mark_chain import MarkovChains
from models.rnn import RNN
from models.lin_reg import LinReg
# trie takes > 600 seconds to load, skipping
# from models.trie import Trie
from preproc.filters import markov_filter, rnn_filterv2


app = Flask(__name__)


def load_pipeline(verbose=False):
    models = [
        ('Difference Table', DiffTable(), None),
        ('Linear Recurrent Relation', LinearModel(minlen=2), None),
        ('Non-linear Recurrent Relation', NonLinearModel(minlen=2), None),
        ('Markov Chain', MarkovChains(6, 20, 4), markov_filter),
        ('Linear Regression', LinReg(), None),
        ('Recurrent NeuralNet', RNN(), rnn_filterv2),
    ]
    pipe = Pipeline(models, fallback=Baseline())
    if verbose:
        print("Pipeline has been loaded now")
    return pipe


model = load_pipeline()


@app.route('/')
@app.route('/index.html', strict_slashes=False)
def entry_point():
    return render_template('index.html')

@app.route('/about.html', strict_slashes=False)
def about_page():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def form_process():
    text = request.form['text']
    # print(text)
    values = prepare_sequence(text)
    if values is not None:
        if any([(x < -10e20) or (x > 10e20) for x in values]):
            return render_template('index.html', text="", prediction=None, pred_by=None, bad_value="Too big absolute value in a sequence")
        modname, pred = model.predict1(values)
        pred = str(int(pred))
        return render_template('index.html', text=text, prediction=pred, pred_by=modname)
    else:
        return render_template('index.html', text=text, prediction=None, pred_by=None)


def prepare_sequence(text):
    try:
        return list(map(int, text.split(',')))
    except ValueError:
        return None


@app.route('/ping', methods=['GET'])
def another():
    return jsonify('hi')

@app.route('/page')
def static_page():
    return render_template('index.html')
