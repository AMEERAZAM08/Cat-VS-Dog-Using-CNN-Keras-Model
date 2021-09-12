import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
model = load_model('model.h5')

st.title('CatDog Classifier')



file = st.file_uploader("Upload Cat or Dog  image", type=["png", "jpg", "jpeg"])

if file is not None:
    image = Image.open(file)

    st.image(
        image,
        caption=f"Uploaded Image",
        use_column_width=True,
    )

    img_array = np.array(image)
    img = tf.image.resize(img_array, size=(64,64))
    img = tf.expand_dims(img, axis=0)
    result = model.predict(img)
    #training_set.class_indices
    
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat' 
    #print(prediction)
    st.write(prediction)
