import google.generativeai as genai

import streamlit as st

genai.configure(api_key="")


model= genai.GenerativeModel("gemini-pro")


chat= model.start_chat(history=[])


def get_res(text):
    res= chat.send_message(text, stream=True)

    return res

input = st.text_input("Query:")
submit= st.button("Submit")

if submit and input:
    res= get_res(input)

    st.subheader("Query: ", input)

    for chunk in res:
        st.write(chunk.text)


