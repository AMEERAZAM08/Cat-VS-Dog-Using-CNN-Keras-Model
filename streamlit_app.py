import streamlit as st
import os
import numpy as np
import time
from keras.preprocessing import image
from keras.models import load_model
import os
model = load_model('model.h5')

st.title('CatDog Classifier')

def file_selector(folder_path='.'):
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select Image file', filenames)
    return os.path.join(folder_path, selected_filename)

filename = file_selector()
st.write('You selected `%s`' % filename)
st.write('Classifying this image...')

if filename is not None and (filename[-4:]=='.jpg' or filename[-4:]=='.png' or filename[-4:]=='.jpeg' or filename[-4:]=='.JPG' or filename[-4:]=='.PNG' or filename[-4:]=='.JPEG'):

    #st.write('Image file selected: ', filename)
    test_image = image.load_img(filename, target_size = (64, 64))
    #print(test_image)
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = model.predict(test_image)
    #training_set.class_indices
    
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
    time.sleep(5)  
    #print(prediction)
    st.write(prediction)
else:
    st.write('Please select an image file')
    st.write('Exiting...')




