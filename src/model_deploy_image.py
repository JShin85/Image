import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

model = load_model("project_model_crawling_img.h5")

CLASS_NAME = ["british shorthair", "munchkin", "ragdoll", "sphynx"]

st.title("반려묘 구분하기")
st.markdown("반려묘의 이미지를 업로드 해주세요")

cat_image = st.file_uploader("이미지 선택", type="jpg")
submit = st.button("구분하기")

if submit:
    if cat_image is not None:
        file_bytes = np.asarray(bytearray(cat_image.read()), dtype = np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)

        st.image(opencv_image, channels="BGR")
        opencv_image = cv2.resize(opencv_image, (224, 224))
        opencv_image.shape = (1, 224, 224, 3)
        Y_pred = model.predict(opencv_image)

        st.title(str("이 반려묘의 품종은 " + CLASS_NAME[np.argmax(Y_pred)]))
