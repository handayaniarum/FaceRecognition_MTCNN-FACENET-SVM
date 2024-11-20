import cv2
import os
import time
from mtcnn.mtcnn import MTCNN

# Nama folder untuk menyimpan gambar wajah yang terdeteksi
dataset_folder = "dataset"  # Folder utama yang berisi folder dataset

# Membuat Folder Nama untuk data muka baru
class_name = input("Input Class Name: ")

# Membuat folder untuk nama individu di dalam `dataset` jika belum ada
class_folder = os.path.join(dataset_folder, class_name)
if not os.path.exists(class_folder):
    os.makedirs(class_folder)
    print(f"[INFO] Folder '{class_name}' created in '{dataset_folder}'")
else:
    print(f"[INFO] Folder '{class_name}' already exists in '{dataset_folder}'")

# Menginisialisasi kamera dengan resolusi tertentu
cap = cv2.VideoCapture(1)

# Inisiasi detektor
detector = MTCNN()

# Mengatur variabel untuk menyimpan gambar dengan urutan
frame_rate = int(input("Input Frame Rate (FPS): "))  # frames per second
duration = 10  # durasi dalam detik (berapa lama loop menangkap sejumlah frame)
start_time = time.time()
num_frames = frame_rate * duration

# Menentukan indeks gambar berikutnya
existing_files = os.listdir(class_folder)
if existing_files:
    # Mengambil indeks terbesar dari file yang sudah ada
    existing_indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files if f.endswith('.jpg')]
    start_index = max(existing_indices) + 1
else:
    start_index = 0

# Loop untuk menangkap frame dari kamera
i = start_index
while i < start_index + num_frames:
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi frame ke RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi wajah (menggunakan MTCNN)
    detections = detector.detect_faces(frame_rgb)
    for det in detections:
        if det['confidence'] >= 0.7:
            x, y, w, h = det['box']
            x, y = max(x, 0), max(y, 0)

            # Crop dan simpan wajah terdeteksi
            face = frame_rgb[y:y + h, x:x + w]
            # Resize wajah ke ukuran 160x160
            face_resized = cv2.resize(face, (160, 160))
            # Konversi RGB ke BGR
            face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

            # Path untuk menyimpan gambar wajah
            filename = f'{class_name}_{i:04}.jpg'
            local_file_path = os.path.join(class_folder, filename)
            cv2.imwrite(local_file_path, face_bgr)
            print(f"[INFO] Saved Local {local_file_path}")

            # Gambar bounding box di frame asli
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            # Gambar landmarks
            for key, point in det['keypoints'].items():
                cv2.circle(frame, point, 2, (0, 255, 0), 2)
                #cv2.putText(frame, key, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Tampilkan frame dengan bounding box jika diperlukan
    cv2.imshow('frame', frame)

    # Menunggu untuk frame berikutnya dengan waktu sesuai frame rate
    time_elapsed = time.time() - start_time
    while time_elapsed < (i + 1 - start_index) / frame_rate:
        time_elapsed = time.time() - start_time

    i += 1  # Menambahkan indeks untuk urutan berikutnya

    # Break loop jika tombol 'q' ditekan
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release kamera dan tutup semua jendela
cap.release()
cv2.destroyAllWindows()
