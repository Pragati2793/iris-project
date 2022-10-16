from flask import Flask, render_template, request
import numpy as np
import pickle

model = pickle.load(open('iri.pkl', 'rb'))


app = Flask(__name__)


@app.route('/')
def man():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def home():
    a = request.form['a']
    b = request.form['b']
    c = request.form['c']
    d = request.form['d']

    arr = np.array([[a, b, c, d]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug=False)   

# @app.route('/predict1')
# def predict1():
#     a = int(request.args.get('a'))
#     b = int(request.args.get('b'))
#     c = int(request.args.get('c'))
#     d = int(request.args.get('d'))

#     arr = np.array([[a, b, c, d]])
#     pred = model.predict(arr)
#     return render_template('after.html', data=pred)
    
