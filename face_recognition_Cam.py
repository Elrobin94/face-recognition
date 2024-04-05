from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import cv2
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
import joblib

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create a HOG detector with customized parameters
    winSize = (64, 64)
    blockSize = (32, 32)  
    blockStride = (16, 16)
    cellSize = (16, 16)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    # Extract the HOG features
    hog_features = hog.compute(gray_image)
    return hog_features

def predict_image_HOG(image, classifier, hog_pca_transformer):
    predict_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Verkleinere das Bild
    predict_resized_image = cv2.resize(predict_image, (255, 255))
    # Extrahiere die HOG-Merkmale
    hog_features = extract_hog_features(predict_resized_image)
    hog_features_pca = hog_pca_transformer.transform(hog_features.reshape(1, -1))
  # Führe die Vorhersage mit dem Klassifikator durch
    probabilities = classifier.predict_proba(hog_features_pca)[0]
    predicted_class_index = np.argmax(probabilities)
    predicted_class_name = classifier.classes_[predicted_class_index]
    confidence = probabilities[predicted_class_index]
    return predicted_class_name, confidence

    
def detect_face(image_path):
    # Laden des Gesichtserkennungs-Klassifikators (Haar-Cascade)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # Laden des Bildes
    image = cv2.imread(image_path)
    if image is not None:
        # Konvertieren des Bildes in Graustufen
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Gesichter im Bild erkennen
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(200, 200))
        if len(faces) > 0:
            # Nehme das erste erkannte Gesicht
            (x, y, w, h) = faces[0]
            # Schneide das Gesicht aus
            face = image[y:y+h, x:x+w]
             # Zeichne ein Rechteck um das erkannte Gesicht
            ##cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            return face
        else:
            print(f"No face detected in {image_path}")
            return None
    else:
        print(f"Failed to load image: {image_path}")
        return None



