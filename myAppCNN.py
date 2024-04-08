import streamlit as st
from PIL import Image
import numpy as np
import joblib
import sys
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten,BatchNormalization, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

# Laden der trainierten Modelle
loaded_model = keras.models.load_model("E:/ZHAW/Semester4/Project/face-recognition/mein_keras_modell.h5")

# Funktion zur Erkennung von Gesichtern in Bildern
def detect_face(image):
    # Laden des Gesichtserkennungs-Klassifikators (Haar-Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Konvertieren des Bildes in ein OpenCV-Array
    frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Konvertieren des Bildes in Graustufen
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Gesichter im Bild erkennen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) > 0:
        # Nehme das erste erkannte Gesicht
        (x, y, w, h) = faces[0]
        # Schneide das Gesicht aus
        face = frame_bgr[y:y+h, x:x+w]
        return Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
    else:
        st.warning("Kein Gesicht erkannt.")
        return None

# Streamlit App
def main():
    st.title("Face Recognition App")

    # Option zur Auswahl des Kamera-Modus oder Datei-Upload
    camera_mode = st.sidebar.checkbox("Kamera-Modus", True)
    
    # Hier können Sie die Kamera steuern und ein Bild aufnehmen
    st.write("Kamera wird aktiviert...")
    # Öffnen Sie die Kamera
    cap = cv2.VideoCapture(0)
    if st.button("Bild aufnehmen"):
        ret, frame = cap.read()
        if ret:
            # Gesichter im Bild erkennen
            detected_face = detect_face(frame)
            if detected_face is not None:
                # Vorhersage auf dem Gesicht durchführen
                image_preprocessed = preprocess_input(np.array(detected_face.resize((224, 224))))
                image_batch = np.expand_dims(image_preprocessed, axis=0)
                predictions = loaded_model.predict(image_batch)
                
                if predictions is not None:
                    class_names = ['robin', 'nino']
                    predicted_class_index = np.argmax(predictions)
                    predicted_class = class_names[predicted_class_index]
                    confidence = predictions[0][predicted_class_index]
                    st.success(f"Vorhersage: {predicted_class} (Vertrauen: {confidence:.2f})")
                else:
                    st.warning("Gesicht nicht erkannt oder Vorhersage fehlgeschlagen.")
            else:
                st.warning("Gesicht nicht erkannt.")
            # Zeige das ursprüngliche Bild an
            st.image(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), caption="Original Image", use_column_width=True)
        else:
            st.warning("Fehler beim Aufnehmen des Bildes.")
    else:
        st.warning("Bitte klicken Sie auf 'Bild aufnehmen', um ein Bild aufzunehmen.")

if __name__ == "__main__":
    main()