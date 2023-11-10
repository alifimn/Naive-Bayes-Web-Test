from flask import Flask, render_template, request
from naive_bayes_model import NaiveBayesClassifier

app = Flask(__name__)

# Inisialisasi model
classifier = NaiveBayesClassifier()

# Rute untuk halaman beranda
@app.route('/')
def home():
    return render_template('index.html')

# Rute untuk menangani prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil input dari formulir
    input_text = [request.form['text']]

    # Gunakan model Naive Bayes untuk membuat prediksi
    prediction = classifier.predict(input_text)

    # Tampilkan hasil prediksi pada halaman web
    return render_template('result.html', prediction=prediction[0])

# ...

if __name__ == '__main__':
    # Melatih model (ganti dengan data dan label sesuai kebutuhan)
    X_train = ["kucing", "andi"]
    y_train = ["hewan", "manusia"]
    classifier.train(X_train, y_train)

    # Menyimpan model
    classifier.save_model()

    # Menjalankan aplikasi Flask
    app.run(debug=True)

