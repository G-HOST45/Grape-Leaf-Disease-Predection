import streamlit as st
import cv2 as cv
import numpy as np
import keras


label_name = ['Grape Black rot', 'Grape Esca', 'Grape Leaf blight', 'Grape healthy']

st.write("""The grape leaf disease detection model is built using deep learning techniques, and it uses transfer learning to leverage the pre-trained knowledge of a base model. The model is trained on a dataset containing images of 33 different types of leaf diseases. For more information about the architecture, dataset, and training process, please refer to the code and documentation provided.""")              

st.write("Please input only leaf Images of Grape Leaves. Otherwise, the model will not work perfectly.")

model = keras.models.load_model('D:\Projects\leaf-diseases-detect-main\Training\model\Leaf Deases(96,88)5.h5')


uploaded_file = st.file_uploader("Upload an image")
if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    img = cv.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv.IMREAD_COLOR)
    normalized_image = np.expand_dims(cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (150, 150)), axis=0)
    predictions = model.predict(normalized_image)
    st.image(image_bytes)

    # if predictions[0][np.argmax(predictions)]*100 >= 80:
    #     st.write(f"Result is : {label_name[np.argmax(predictions)]}")
    
    # else:st.write(f"Try Another Image")
    pred = predictions[0]  # get the first prediction row
    idx = np.argmax(pred)  # index of the highest confidence
    confidence = pred[idx] * 100
    if confidence >= 80:
        if idx < len(label_name):
            st.write(f"Result is : {label_name[idx]} ({confidence:.2f}%)")
        else:
            st.write(f"Error: Prediction index {idx} out of range.")
    else:
        st.write("Try Another Image")