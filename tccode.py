import streamlit as st 
import tensorflow as tf
from PIL import Image
import numpy as np
st.set_page_config(page_title="Toyota Crolla Parts Clasifier" ,page_icon= "🚗")
st.title("Toyota Crolla Parts Clasifier 🚗🛞")
st.header("fresh vs rotten ")

model = tf.keras.models.load_model('tcmodel.h5')


upload_image = st.file_uploader("select image" , type=['jpg' , 'jpeg' ,'png'])
if upload_image is not None :
    image= Image.open(upload_image)
    st.image(image , caption = "image uploader" )


    if image.mode != "RGB":
     image = image.convert("RGB")
    
    image = image.resize((64,64))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image,axis = 0)


    pred = model.predict(image)
    class_index = np.argmax(pred)
    class_list = ["Air Filter","Battery","Coolant Reservoir","Engine","Engine Cover","Reservoir Cap"
             ,"Windshield Fluid"]    
    btn = st.button("Print Result")
    if btn :
     st.success(class_list[class_index])