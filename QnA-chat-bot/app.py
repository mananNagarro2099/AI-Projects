from dotenv import load_dotenv
load_dotenv() ## this is basically meant for loading all the env variables
import streamlit as st
import os
import google.generativeai as genai
from PIL import Image


genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



gemini_model = genai.GenerativeModel("gemini-1.5-pro")
gemini_image_model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(input=None,image=None):
    if input is None or input == "":
        print("using gemini_image_model")
        response = gemini_image_model.generate_content(image)
    if image is None:
        print("using gemini_model")
        response = gemini_model.generate_content(input)
    else:
        print("using gemini_image_model")
        response = gemini_image_model.generate_content([input,image])
    return response.text




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
        st.subheader("Response:")
        st.write(response)


# models/gemini-1.0-pro-vision-latest
# models/gemini-pro-vision
# models/gemini-1.5-pro-latest
# models/gemini-1.5-pro-001
# models/gemini-1.5-pro-002
# models/gemini-1.5-pro
# models/gemini-1.5-flash-latest
# models/gemini-1.5-flash-001
# models/gemini-1.5-flash-001-tuning
# models/gemini-1.5-flash
# models/gemini-1.5-flash-002
# models/gemini-1.5-flash-8b
# models/gemini-1.5-flash-8b-001
# models/gemini-1.5-flash-8b-latest
# models/gemini-1.5-flash-8b-exp-0827
# models/gemini-1.5-flash-8b-exp-0924
# models/gemini-2.5-pro-exp-03-25
# models/gemini-2.5-pro-preview-03-25
# models/gemini-2.5-flash-preview-04-17
# models/gemini-2.5-flash-preview-05-20
# models/gemini-2.5-flash-preview-04-17-thinking
# models/gemini-2.5-pro-preview-05-06
# models/gemini-2.0-flash-exp
# models/gemini-2.0-flash
# models/gemini-2.0-flash-001
# models/gemini-2.0-flash-exp-image-generation
# models/gemini-2.0-flash-lite-001
# models/gemini-2.0-flash-lite
# models/gemini-2.0-flash-preview-image-generation
# models/gemini-2.0-flash-lite-preview-02-05
# models/gemini-2.0-flash-lite-preview
# models/gemini-2.0-pro-exp
# models/gemini-2.0-pro-exp-02-05
# models/gemini-exp-1206
# models/gemini-2.0-flash-thinking-exp-01-21
# models/gemini-2.0-flash-thinking-exp
# models/gemini-2.0-flash-thinking-exp-1219
# models/gemini-2.5-flash-preview-tts
# models/gemini-2.5-pro-preview-tts
# models/learnlm-2.0-flash-experimental