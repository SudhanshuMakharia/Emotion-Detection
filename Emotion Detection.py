import os
import sys
import random
import itertools
import shutil
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
from tempfile import NamedTemporaryFile

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import mediapipe as mp
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

# Constants
CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = 'emotion-detection-fer:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-data-sets%2F1028436%2F1732825%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240715%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240715T080251Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D4539041b4d3f6efb4a9d1a64502599f255dade85e0e20c4183573d8c91785ff6b8cdbd4824457da7e4e0b468c9f3fd7bc5f81627d8b167caea5c3dc6c62507aaea58728b55aa36dc7ca4be3454e37da0b6a5bb17ee96fa8cc4389afe707e5f1afce604c7515086c825cd3bbcce30e1a126bb3688026569e21e62e43702219c0ef7e006a806845e8300334a23258406598524bdf0ece349d0e491a0de439d5d35b78407eedad1f21f9921e70e4e17d87b0760282d837a37891e0ec468e812fc9bcdd0035dc189c87eeb79ce68a35454f69c35a630b25d9453fbe243c277d932e0c1f05ee0801fd8c0a94e116ddb8b6a4aca33877edf3fcc1cd785623d1f164a6d'
KAGGLE_INPUT_PATH = 'input'
KAGGLE_WORKING_PATH = 'working'

# Create directories
os.makedirs(KAGGLE_INPUT_PATH, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, exist_ok=True)

# Download and extract data
for data_source_mapping in DATA_SOURCE_MAPPING.split(','):
    directory, download_url_encoded = data_source_mapping.split(':')
    download_url = unquote(download_url_encoded)
    filename = os.path.basename(urlparse(download_url).path)
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = int(fileres.headers['content-length'])
            print(f'Downloading {directory}, {total_length} bytes compressed')
            dl = 0
            while True:
                data = fileres.read(CHUNK_SIZE)
                if not data:
                    break
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / total_length)
                sys.stdout.write(f"\r[{'=' * done}{' ' * (50 - done)}] {dl} bytes downloaded")
                sys.stdout.flush()
            tfile.seek(0)
            if filename.endswith('.zip'):
                with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
            else:
                with tarfile.open(fileobj=tfile) as tarfile_obj:
                    tarfile_obj.extractall(destination_path)
            print(f'\nDownloaded and uncompressed: {directory}')
    except HTTPError as e:
        print(f'Failed to load (likely expired) {download_url} to path {destination_path}')
    except OSError as e:
        print(f'Failed to load {download_url} to path {destination_path}')

print('Data source import complete.')

# Install mediapipe
#pip install mediapipe

# Load data and process
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
LEFT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYE)))
RIGHT_EYE = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYE)))
LEFT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LEFT_EYEBROW)))
RIGHT_EYEBROW = list(set(itertools.chain(*mp_face_mesh.FACEMESH_RIGHT_EYEBROW)))
LIPS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_LIPS)))
CONTOURS = list(set(itertools.chain(*mp_face_mesh.FACEMESH_CONTOURS)))
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

def euc2d(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def euc3d(a, b):
    return np.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)

emotions = os.listdir(os.path.join(KAGGLE_INPUT_PATH, 'emotion-detection-fer/train'))
face_features = pd.DataFrame({}, columns=[f"{i}" for i in range(92 * 2)] + ["y"])

for i, emotion in enumerate(emotions):
    images = os.listdir(os.path.join(KAGGLE_INPUT_PATH, 'emotion-detection-fer/train', emotion))
    selected_images = random.sample(images, 100)
    for image in selected_images:
        img_path = os.path.join(KAGGLE_INPUT_PATH, 'emotion-detection-fer/train', emotion, image)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
        results = face_mesh.process(img)
        if results.multi_face_landmarks:
            shape = [(lmk.x, lmk.y, lmk.z) for lmk in results.multi_face_landmarks[0].landmark]
            shape = np.array(shape)
            nose = shape[1]
            shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS]
            distances2d = [round(euc2d(nose, x), 6) for x in shape]
            distances3d = [round(euc3d(nose, x), 6) for x in shape]
            face_features.loc[len(face_features)] = distances2d + distances3d + [i]

face_features = shuffle(face_features)
X = face_features.iloc[:, :-1].values
y = face_features.iloc[:, -1].values
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
y_train = to_categorical(y)
X_train = X_train[..., np.newaxis]

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(len(emotions), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20)

# Streamlit app
st.title("Emotion Recognition from Facial Expressions")

# Capture image using the camera
st.write("Click the button below to capture an image using your device's camera.")
img_file_buffer = st.camera_input("Capture an image")

if img_file_buffer:
    # Read the image from buffer
    img = cv2.imdecode(np.frombuffer(img_file_buffer.read(), np.uint8), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image and make predictions
    results = face_mesh.process(img_rgb)
    if results.multi_face_landmarks:
        shape = [(lmk.x, lmk.y, lmk.z) for lmk in results.multi_face_landmarks[0].landmark]
        shape = np.array(shape)
        nose = shape[1]
        shape = shape[LEFT_EYE + RIGHT_EYE + LEFT_EYEBROW + RIGHT_EYEBROW + LIPS]
        distances2d = [round(euc2d(nose, x), 6) for x in shape]
        distances3d = [round(euc3d(nose, x), 6) for x in shape]
        input_features = distances2d + distances3d
        input_features = scaler.transform([input_features])
        input_features = input_features[..., np.newaxis]
        
        prediction = model.predict(input_features)
        predicted_emotion = emotions[np.argmax(prediction)]
        
        st.image(img_rgb, caption=f"Predicted emotion: {predicted_emotion}")
    else:
        st.write("No face detected. Please try again.")
else:
    st.write("No image captured.")
