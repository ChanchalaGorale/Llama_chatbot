import google.generativeai as genai
import streamlit as st
from PIL import Image


# configure genai with api key

genai.configure(api_key="AIzaSyC0HQTg-oaShAG_0_GgxIqUCTTjtqRRK9E")

# set up model for text
model = genai.GenerativeModel("gemini-pro-vision")


def get_res(text, image):
    if text !="":

        res= model.generate_content([text, image])

    else: 
        res= model.generate_content(image)

    return res.text

st.header("Gemeini Text API ")

input = st.text_input("Enter your query")
file = st.file_uploader("Upload image")
submit = st.button("Ask Question")
image = ""

if file:
    image = Image.open(file)
    st.image(image, use_column_width=True)




if submit and image:
    res = get_res(input, image)
    st.subheader("Response:")
    st.write(res)
   
        





























