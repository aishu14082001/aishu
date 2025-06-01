import streamlit as st
import face_recognition
import numpy as np
from PIL import Image
import os

st.title("🧠 Childhood Photo Matcher")
st.write("Upload a childhood photo to guess the team member!")

uploaded_file = st.file_uploader("Upload a photo", type=["jpg", "png"])

# Load training data
known_face_encodings = []
known_face_names = []

for name in os.listdir("dataset"):
    for img_path in os.listdir(f"dataset/{name}"):
        img = face_recognition.load_image_file(f"dataset/{name}/{img_path}")
        encoding = face_recognition.face_encodings(img)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(name)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Childhood Photo", use_column_width=True)

    test_image = np.array(image)
    encodings = face_recognition.face_encodings(test_image)

    if encodings:
        distances = face_recognition.face_distance(known_face_encodings, encodings[0])
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.6:
            st.success(f"🎉 This is most likely: **{known_face_names[best_match_index]}**")
        else:
            st.warning("❌ No matching team member found.")
    else:
        st.error("😕 No face detected in the image.")
