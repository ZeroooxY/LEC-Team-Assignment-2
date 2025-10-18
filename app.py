from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load model CNN hasil optimasi (utuh)
MODEL_PATH = 'optimized_model.h5'
model = load_model(MODEL_PATH)

# Kelas sesuai dataset (CIFAR-10)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="Tidak ada file yang diunggah")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', prediction="File belum dipilih")

    # Simpan sementara
    file_path = os.path.join('static', file.filename)
    file.save(file_path)

    # Preprocess gambar
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100

    result = f"{predicted_class} ({confidence:.2f}% confidence)"
    return render_template('index.html', prediction=result, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
