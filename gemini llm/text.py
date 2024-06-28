import google.generativeai as genai
import streamlit as st


# configure genai with api key

genai.configure(api_key="AIzaSyC0HQTg-oaShAG_0_GgxIqUCTTjtqRRK9E")

# set up model for text
model = genai.GenerativeModel("gemini-pro")

def get_res(text):
    res= model.generate_content(text, stream=True)

    return res

st.header("Gemeini Text API ")


input = st.text_input("Enter your query")
submit = st.button("Ask Question")

if submit and input:
    res = get_res(input)
    st.subheader("Response:")
    for c in res:
        txt =c.text
        st.write(txt)





























