import numpy as np
import json
import prepare
from flask import Flask, render_template
from flask import request
from PIL import Image

INPUT = 784
HIDDEN = 392
OUT = 10
IMAGE_NUMBER = 1

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/request', methods=['POST', 'GET'])
def test():
    output = request.get_json()
    result = list(json.loads(output)['data'].values())[3::4]
    im = Image.new('L', (500, 500), 255)
    imgdata = im.load()
    for x in range(im.size[0]):
        for y in range(im.size[1]):
            imgdata[x, y] = 255 - result[y * 500 + x]
    im.save('image.png')
    img = prepare.rec_digit('image.png')
    prediction = predict(np.array(img).reshape(1, 784))
    number = np.argmax(prediction)
    print(number)
    return str(number)


def relu(x):
    return np.maximum(x, 0)


def softmax(x):
    out = np.exp(x)
    return out / np.sum(out)


def predict(x):
    t1 = x @ w1 + b1
    h1 = relu(t1)
    t2 = h1 @ w2 + b2
    z = softmax(t2)
    return z


if __name__ == "__main__":
    with open('mnistTrain.json', 'r') as digits:
        data = json.load(digits)
        w1 = np.array(data['layers'][1]['weights']).transpose()
        w2 = np.array(data['layers'][2]['weights']).transpose()
        b1 = np.array(data['layers'][1]['biases'])
        b2 = np.array(data['layers'][2]['biases'])
    app.run()

