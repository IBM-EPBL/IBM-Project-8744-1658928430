from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.models import Sequential
app = Flask(__name__)


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/upload')
def upload_file2():
    return render_template('web.html')


@app.route('/predict', methods=['POST'])
def upload_image_file():
    if request.method == 'POST':
        model = load_model(r'models/mnistCNN.h5')
        # print("Files-------------")
        # print("------------------")
        # print(request.files.keys[0])
        # print("------------------")
        # print("------------------")
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 1)
        y_pred = model.predict(im2arr)
        result = np.argmax(y_pred, axis=1)
        print(result)

        if (result == 0):
            return render_template("0.html", showcase=str(result))
        elif (result == 1):
            return render_template("1.html", showcase=str(result))
        elif (result == 2):
            return render_template('2.html', showcase=str(result))
        elif (result == 3):
            return render_template('4.html', showcase=str(result))
        elif (result == 4):
            return render_template('5.html', showcase=str(result))
        elif (result == 6):
            return render_template('6.html', showcase=str(result))
        elif (result == 7):
            return render_template('7.html', showcase=str(result))
        elif (result == 8):
            return render_template('8.html', showcase=str(result))
        else:
            return render_template('9.html', showcase=str(result))
    else:
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
