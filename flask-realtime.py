from flask import Flask, Response
import cv2 as cv
import numpy as np
import joblib
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from scipy.special import expit
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, precision_score, recall_score
from collections import deque
import time
import requests
url = 'https://presensi.sindo.ro/api/markAttendance'

app = Flask(__name__)

# Konstanta dan Inisialisasi
CONFIDENCE_THRESHOLD = 0.84
SIMILARITY_THRESHOLD = 0.7
FRAME_THRESHOLD = 2 # 3&10: buffer harus ngisi 3 wajah konsisten dari 10 frame (history_size)
HISTORY_SIZE = 10

svm_model_filename = 'svm_face_recognition_model80.pkl'
encoder_filename = 'label_encoder80.pkl'
embedder = FaceNet()
model = joblib.load(svm_model_filename)
encoder = joblib.load(encoder_filename)
detector = MTCNN()

# Metrik performa
true_labels = []
predicted_labels = []
frame_count = 0
# Variabel untuk menyimpan true_labels dan predicted_labels yang baru
batch_true_labels = []
batch_predicted_labels = []

# Variabel untuk menghitung FPS
fps_start_time = time.time()
fps_frame_count = 0

def get_embedding(face_img):
    try:
        face_img = face_img.astype('float32')
        face_img = np.expand_dims(face_img, axis=0)
        embedding = embedder.embeddings(face_img)[0]
        return embedding
    except Exception as e:
        print("Error getting embedding:", e)
        raise

def get_prediction_confidence(embedding):
    decision = model.decision_function([embedding])[0]
    confidence = expit(decision)[0]
    return confidence

def wrap_text(text, font, font_scale, thickness, max_width):
    words = text.split(' ')
    lines = []
    current_line = words[0]

    for word in words[1:]:
        width, _ = cv.getTextSize(current_line + ' ' + word, font, font_scale, thickness)[0]
        if width <= max_width:
            current_line += ' ' + word
        else:
            lines.append(current_line)
            current_line = word

    lines.append(current_line)
    return lines

def get_consistent_face(face_id):
    if face_id in detected_faces:
        face_buffer = detected_faces[face_id]
        if len(face_buffer) >= FRAME_THRESHOLD:
            consistent_face = max(face_buffer, key=lambda f: f['confidence'])
            if consistent_face['confidence'] > CONFIDENCE_THRESHOLD:
                return consistent_face['name']
    return "unknown"

def calculate_and_print_metrics():
    if true_labels and predicted_labels:
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)

        print(f"Akurasi: {accuracy:.2f}")
        print(f"Presisi: {precision:.2f}")
        print(f"Recall: {recall:.2f}")

detected_faces = {}
stored_embeddings = []
processed_faces_in_frame = set()
recognized_names = set()
consistent_names = set()

def generate_frames():
    global frame_count
    global fps_frame_count
    global fps_start_time
    global batch_true_labels
    global batch_predicted_labels
    global true_labels
    global predicted_label

    cap = cv.VideoCapture(1)  # Use 0 for the default webcam
    while True:
        start_time = time.time()  # Mulai waktu untuk keseluruhan frame
        success, frame = cap.read()
        if not success:
            break

        # Mengukur waktu deteksi wajah
        detect_start = time.time()
        gray_img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized_img = clahe.apply(gray_img)
        rgb_img = cv.cvtColor(equalized_img, cv.COLOR_GRAY2RGB)
        faces = detector.detect_faces(rgb_img)
        detect_end = time.time()
        detect_time = detect_end - detect_start
        print(f"Waktu deteksi: {detect_time:.4f} seconds")
        current_frame_faces = []
        processed_faces_in_frame.clear()

        for face in faces:
            x, y, w, h = face['box']
            x, y = max(x, 0), max(y, 0)
            center_x, center_y = x + w // 2, y + h // 2

            face_id = (center_x, center_y)

            if face_id in processed_faces_in_frame:
                continue

            processed_faces_in_frame.add(face_id)

            # Mengukur waktu ekstraksi embedding
            embedding_start = time.time()
            face_img = rgb_img[y:y + h, x:x + w]
            face_img = cv.resize(face_img, (160, 160))

            embedding = get_embedding(face_img)
            is_recognized = False
            face_name = "unknown"
            embedding_end = time.time()
            embedding_time = embedding_end - embedding_start
            print(f"Waktu ekstraksi embedding: {embedding_time:.4f} seconds")

            # Mengukur waktu prediksi
            prediction_start = time.time()

            for stored_embedding in stored_embeddings:
                similarity = cosine_similarity([embedding], [stored_embedding['embedding']])[0][0]

                # Cek apakah similarity di atas threshold untuk dianggap sebagai wajah yang sudah ada
                if similarity > SIMILARITY_THRESHOLD:
                    face_name = stored_embedding['name']
                    is_recognized = True
                    break

            if not is_recognized:
                # Jika tidak ada wajah yang cocok dalam threshold similarity, prediksi sebagai "unknown"
                face_label = model.predict([embedding])[0]
                face_name = encoder.inverse_transform([face_label])[0]

                # Hanya simpan jika confidence cukup tinggi
                confidence = get_prediction_confidence(embedding)
                if confidence > CONFIDENCE_THRESHOLD and face_name != "unknown":
                    stored_embeddings.append({'name': face_name, 'embedding': embedding})
                    print(f"Nama baru disimpan: {face_name}")
                else:
                    face_name = "unknown"
                    print("Wajah tidak dikenali, dilabel sebagai 'unknown'")

            prediction_end = time.time()
            prediction_time = prediction_end - prediction_start
            print(f"Waktu prediksi: {prediction_time:.4f} seconds")

            confidence = get_prediction_confidence(embedding)
            if confidence < CONFIDENCE_THRESHOLD:
                face_name = "unknown"
                confidence_text = ""
            else:
                confidence_text = f"{confidence:.2f}"

            if face_name != "unknown":
                recognized_names.add(face_name)
                print(f"Nama yang dikenali: {face_name}")

            # Pembaruan true_labels dan predicted_labels
            true_label = "Handayani Arum"  # Label yang tersedia
            predicted_label = face_name
            true_labels.append(true_label)
            predicted_labels.append(predicted_label)

            current_frame_faces.append(
                {'face_id': face_id, 'name': face_name, 'confidence': confidence, 'box': (x, y, w, h)})

        for face in current_frame_faces:
            face_id = face['face_id']
            if face_id not in detected_faces:
                detected_faces[face_id] = deque(maxlen=HISTORY_SIZE)

            detected_faces[face_id].append(face)

            if len(detected_faces[face_id]) >= FRAME_THRESHOLD:
                consistent_name = get_consistent_face(face_id)
                if consistent_name != "unknown":
                    consistent_names.add(consistent_name)
                    print(f"Nama konsisten: {consistent_name}")

                    # Mengukur waktu pengiriman data ke API
                    api_start = time.time()
                    objek = {
                        'name': consistent_name
                    }

                    x = requests.post(url, json=objek)
                    api_end = time.time()
                    api_time = api_end - api_start
                    print(f"Waktu API request: {api_time:.4f} seconds")

                    print(x.text)

        # Gambar bounding box dan label untuk setiap wajah yang terdeteksi
        for face in current_frame_faces:
            x, y, w, h = face['box']
            face_name = face['name']
            confidence_text = f"{face['confidence']:.2f}" if face['confidence'] > CONFIDENCE_THRESHOLD else ""

            # Tambahkan "berhasil" jika nama sudah konsisten
            if face_name in consistent_names:
                label_text = f"{face_name}"
                status_text = "Berhasil"
            else:
                label_text = face_name
                status_text = ""

            wrapped_text = wrap_text(label_text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1, w)
            if confidence_text:
                wrapped_text.append(confidence_text)

            # Tambahkan teks "Berhasil" di bawah confidence jika sesuai
            if status_text:
                wrapped_text.append(status_text)

            text_height = len(wrapped_text) * 20

            cv.rectangle(frame, (x, y), (x + w, y + h), (9, 9, 182), 2)
            cv.rectangle(frame, (x, y + h), (x + w, y + h + text_height + 20), (9, 9, 182), -1)
            for i, line in enumerate(wrapped_text):
                text_size = cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
                text_x = x + (w - text_size[0]) // 2
                y_pos = y + h + 20 + (i * 20)

                # Jika baris teks adalah "Berhasil", ubah warna menjadi hijau
                if line == "Berhasil":
                    cv.putText(frame, line, (text_x, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)
                else:
                    cv.putText(frame, line, (text_x, y_pos), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1,
                               cv.LINE_AA)

        # Tampilkan nama konsisten tanpa redundansi
        print("Nama-nama yang berhasil dikenali tanpa redundansi:")
        for name in consistent_names:
            print(name)

        # Menghitung FPS
        end_time = time.time()
        frame_rate = 1.0 / (end_time - start_time)
        fps_frame_count += 1
        elapsed_time = end_time - fps_start_time
        if elapsed_time > 1.0:
            fps = fps_frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            fps_frame_count = 0
            fps_start_time = end_time

        # Mengukur waktu keseluruhan untuk memproses frame
        end_time = time.time()
        total_frame_time = end_time - start_time
        print(f"Total waktu pemrosesan frame: {total_frame_time:.4f} seconds")

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Tampilkan metrik performa secara berkala
        frame_count += 1
        if frame_count % 10 == 0:  # Tampilkan setiap 10 frame
            if true_labels and predicted_labels:
                print(f"True Labels (batch): {true_labels}")
                print(f"Predicted Labels (batch): {predicted_labels}")
                calculate_and_print_metrics()
                true_labels.clear()
                predicted_labels.clear()

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Real-Time Face Recognition Attendance System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f4f4f4;
            }
            header {
                background-color: #6EACDA;
                color: #fff;
                padding: 10px 0;
                text-align: center;
                position: fixed;
                width: 100%;
                height: 45px;
                top: 0;
            }
            header img {
                height: 50px; /* Set height for logo */
                width: auto;
                position: absolute;
                left: 20px; /* Align logo to the left */
                top: 5px; /* Align logo vertically in the middle */
            }
            footer {
                background-color: #03346E;
                color: #fff;
                padding: 10px 0;
                text-align: center;
                position: fixed;
                width: 100%;
                height: 45px;
                bottom: 0;
            }
            .container {
                padding: 60px;
                text-align: center;
            }
            img {
                border: 3px solid #6EACDA;
                border-radius: 8px;
                width: 1000px; /* Adjust the width as needed */
                height: 500px; /* Adjust the height as needed */
            }
        </style>
    </head>
    <body>
        <header>
            <img src="static/logo.png" alt="Logo"> <!-- Logo image in the header -->
        </header>
        <div class="container">
            <h2>Real-Time Face Recognition Attendance System</h2>
            <img src="/video_feed" alt="Live Video Feed" />
        </div>
        <footer>
            <p>&copy; 2024 Electrical Engineering Facial Recognition Attendance System. Diponegoro University.</p>
        </footer>
    </body>
    </html>
    '''

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
