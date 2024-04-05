import streamlit as st
from PIL import Image
import numpy as np
import joblib
import sys
from face_recognition_Cam import predict_image_HOG


# Laden der trainierten Modelle und Transformer
# Hier müssten Sie Ihre trainierten Modelle und Transformer laden
rf_classifier_hog_path = "rf_classifier_hog.pkl"
hog_pca_transformer_path = "hog_pca_transformer.pkl"

hog_pca_transformer = joblib.load(hog_pca_transformer_path)
rf_classifier_hog = joblib.load(rf_classifier_hog_path)

# Funktion zum Vorhersagen auf einem Bild
def predict_on_image(image, classifier, transformer):
    # Führen Sie hier die Vorhersage mit Ihrem Modell durch
    prediction, confidence = predict_image_HOG(image, classifier, transformer)
    return prediction, confidence

# Streamlit App
def main():
    st.title("Face Recognition App")

    # Option zur Auswahl des Kamera-Modus oder Datei-Upload
    camera_mode = st.sidebar.checkbox("Kamera-Modus", True)
    if camera_mode:
        # Hier können Sie die Kamera steuern und ein Bild aufnehmen
        st.write("Kamera wird aktiviert...")
        if st.button("Bild aufnehmen"):
            # capture_image_from_camera()  # Beispielcode, um ein Bild von der Kamera aufzunehmen
            # Hier wird das aufgenommene Bild angezeigt
            # image = captured_image
            # st.image(image, caption="Captured Image", use_column_width=True)
            st.warning("Kamera-Modus ist noch nicht implementiert.")
            return
        else:
            st.warning("Bitte klicken Sie auf 'Bild aufnehmen', um ein Bild aufzunehmen.")
            return
    else:
        uploaded_image = st.sidebar.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])
        if uploaded_image is not None:
            # Bild anzeigen
            image = Image.open(uploaded_image)
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            st.warning("Bitte laden Sie ein Bild hoch oder aktivieren Sie den Kamera-Modus.")
            return

    # Vorhersage auf dem Bild durchführen
    prediction, confidence = predict_on_image(image, rf_classifier_hog, hog_pca_transformer)
    
    # Vorhersageergebnis anzeigen
    if prediction is not None:
        st.success(f"Prediction: {prediction}, Confidence: {confidence}")
    else:
        st.warning("Gesicht nicht erkannt oder Vorhersage fehlgeschlagen.")

if __name__ == "__main__":
    main()