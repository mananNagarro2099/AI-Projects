from dotenv import load_dotenv
load_dotenv() ## this is basically meant for loading all the env variables
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



gemini_model = genai.GenerativeModel("gemini-1.5-pro")
gemini_image_model = genai.GenerativeModel("gemini-1.5-flash")
chat = gemini_model.start_chat(history=[])




def get_gemini_response(input=None,image=None):
    if input is None or input == "":
        print("using gemini_image_model")
        response = chat.send_message(input,stream=True)
    if image is None:
        print("using gemini_model")
        response = chat.send_message(input,stream=True)
    else:
        print("using gemini_image_model")
        response = chat.send_message(input,stream=True)
    return response

if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []



# Streamlit UI
st.set_page_config(page_title="Gemini Demo")
st.header("Gemini LLM Application")

# User input
input = st.text_input("Enter your question:", key="input")
uploaded_file = st.file_uploader("Upload an image (optional)", type=["jpg", "jpeg", "png"])

# Process the image file if uploaded
image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file)


# Submit button
if st.button("Ask Question"):
    with st.spinner("Generating response..."):
        response = get_gemini_response(input, image)
        st.session_state['chat_history'].append(('You : ',input))
        st.subheader("Response:")
        for chunk in response:
            st.write(chunk.text)
            st.session_state['chat_history'].append(('Bot : ',chunk.text))

st.subheader('The Chat History is :')

for role,text in st.session_state['chat_history']:
    st.write(f"{role}{text}")