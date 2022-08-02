import streamlit as st
import numpy as np
import pickle
import cv2


model = pickle.load(open('model_1.pkl', 'rb'))
test_x = pickle.load(open('test_x.pkl', 'rb'))
test_yc = pickle.load(open('test_yc.pkl', 'rb'))

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

st.header('CNN Model on CIFAR10 Data for Image Classification')
st.text(' ')
index = st.slider('Select index number from testing dataset : ',1,900,400)
st.write('Selected Image : ')
st.image(test_x[index])
img = test_x[index]
img = cv2.resize(img, (80,80))
st.image(img)
st.write('Actual label for the image = ',classes[np.argmax(test_yc[index])] )

#predict class
predicted_val = model.predict(np.array([test_x[index]]))
cls_ind = np.argmax(predicted_val)
percentage = np.max(predicted_val)
st.write('Predicted label for the image = ',classes[cls_ind] )
st.write('Percentage chances that the selected image is ', classes[cls_ind], ' = ', percentage, '%')

