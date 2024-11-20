from time import time
import numpy as np
import os
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model FaceNet
facenet = FaceNet()

# Path ke direktori root gambar yang berisi sub-direktori gambar
path = "dataset50/"

X = []  # List untuk menyimpan embeddings
Y = []  # List untuk menyimpan label
target_names = []  # Array untuk menyimpan nama-nama orang

# Load gambar dan label dari direktori
image_paths = []
labels = []

for directory in os.listdir(path):
    for file in os.listdir(os.path.join(path, directory)):
        image_path = os.path.join(path, directory, file)
        image_paths.append(image_path)
        labels.append(directory)
    target_names.append(directory)

print("Jumlah sampel:", len(image_paths))
print("Kelas:", target_names)

# Encode label
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

# Simpan LabelEncoder
encoder_filename = 'label_encoder80.pkl'
joblib.dump(label_encoder, encoder_filename)
print(f"Encoder disimpan ke {encoder_filename}")

# Pisahkan ke dalam set pelatihan dan pengujian
image_paths_train, image_paths_test, labels_train, labels_test = train_test_split(image_paths, labels_encoded, test_size=0.20, random_state=42)

# Pengaturan augmentasi
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fungsi untuk menghasilkan embedding dari gambar asli
def generate_embeddings_with_augmentation(image_paths, facenet, datagen, augment=True):
    embeddings = []
    labels = []

    for image_path in image_paths:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = np.expand_dims(img_rgb, axis=0)
        label = os.path.basename(os.path.dirname(image_path))  # Menggunakan nama folder sebagai label

        if augment:
            # Augmentasi gambar
            aug_iter = datagen.flow(img_rgb, batch_size=1)
            for _ in range(5):  # Misalnya, kita buat 5 augmentasi untuk setiap gambar
                aug_img = next(aug_iter)[0].astype('uint8')
                aug_img_rgb = cv2.cvtColor(aug_img, cv2.COLOR_BGR2RGB)
                aug_embedding = facenet.embeddings(np.expand_dims(aug_img_rgb, axis=0))[0]
                embeddings.append(aug_embedding)
                labels.append(label)  # Simpan label untuk gambar augmented
        else:
            # Hanya embedding dari gambar asli
            embedding = facenet.embeddings(img_rgb)[0]
            embeddings.append(embedding)
            labels.append(label)

    return np.array(embeddings), np.array(labels)


# Generate embeddings untuk set pelatihan dan pengujian
X_train, y_train = generate_embeddings_with_augmentation(image_paths_train, facenet, train_datagen, augment=True)
X_test, y_test = generate_embeddings_with_augmentation(image_paths_test, facenet, train_datagen, augment=False)

# Encode label
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Menentukan jumlah split untuk cross-validation
min_samples = min(np.bincount(y_train_encoded))
n_splits = min(min_samples, 10)  # Memastikan n_splits tidak melebihi jumlah sampel di kelas terkecil

# Latih classifier SVM dengan embedding yang diperbesar
print("Melatih classifier pada set pelatihan")
t0 = time()

# Define the parameter grid
param_grid = {'C': [0.01, 0.1, 1],
              'gamma': [0.01, 0.1, 1],
              'kernel': ['linear']}
# Initialize GridSearchCV
grid_search = GridSearchCV(SVC(class_weight='balanced'), param_grid,
                           cv=StratifiedKFold(n_splits=n_splits), verbose=2)
# Fit GridSearchCV
grid_search.fit(X_train, y_train_encoded)

# Get best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best params: {best_params}")
print(f"Best score: {best_score}")

# Latih classifier menggunakan data yang diperbesar
clf = SVC(kernel=best_params['kernel'], C=best_params['C'], gamma=best_params['gamma'], class_weight='balanced')

# Latih model dengan data training
clf.fit(X_train, y_train_encoded)

print("Selesai dalam %0.3fs" % (time() - t0))

# Simpan model SVM yang sudah dilatih
model_filename = 'svm_face_recognition_model80.pkl'
joblib.dump(clf, model_filename)
print(f"Model disimpan ke {model_filename}")

# Prediksi pada set uji
print("Memprediksi nama orang pada set uji")
t0 = time()

# Prediksi pada data uji
y_pred = clf.predict(X_test)

print("Selesai dalam %0.3fs" % (time() - t0))

# Konversi label yang diprediksi kembali ke nama asli
y_pred_names = label_encoder.inverse_transform(y_pred)
y_true_names = label_encoder.inverse_transform(y_test_encoded)

# Hitung akurasi pada data pelatihan dan data pengujian
train_accuracy = clf.score(X_train, y_train_encoded)
test_accuracy = clf.score(X_test, y_test_encoded)

# Cetak laporan klasifikasi dan matriks kebingungan
print(classification_report(y_true_names, y_pred_names, target_names=target_names))

print(f"Akurasi pada data pelatihan: {train_accuracy * 100:.2f}%")
print(f"Akurasi pada data pengujian: {test_accuracy * 100:.2f}%")

# Plotting akurasi pelatihan dan pengujian
plt.figure(figsize=(12, 6))

# Subplot: Akurasi Pelatihan dan Pengujian
plt.subplot(1, 2, 1)
plt.bar(['Train Accuracy', 'Test Accuracy'], [train_accuracy * 100, test_accuracy * 100], color=['blue', 'orange'])
plt.title('Training vs Testing Accuracy')
plt.ylabel('Accuracy (%)')
plt.ylim([0, 100])

# Matriks Kebingungan
# Subplot: Matriks Kebingungan
plt.subplot(1, 2, 2)
def plot_confusion_matrix(cm, classes, normalize=False, title='Matriks Kebingungan', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Label yang Benar')
    plt.xlabel('Label yang Diprediksi')

# Hitung matriks kebingungan
cnf_matrix = confusion_matrix(y_true_names, y_pred_names, labels=target_names)

# Plot matriks kebingungan yang tidak dinormalisasi
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=False, title='Matriks Kebingungan')

plt.figure(figsize=(8, 8))
# Plot matriks kebingungan yang dinormalisasi
plot_confusion_matrix(cnf_matrix, classes=target_names, normalize=True, title='Matriks Kebingungan yang Dinormalisasi')

plt.show()