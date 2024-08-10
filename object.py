import streamlit as st
import google.generativeai as genai
from PIL import Image
from gtts import gTTS
import os
import tempfile

api_key = os.getenv('GOOGLE_API_KEY')

# Configure Gemini API
genai.configure(api_key=api_key)

# Set up the model
model = genai.GenerativeModel('gemini-1.5-flash')

def get_gemini_response(image, prompt):
    response = model.generate_content([prompt, image])
    return response.text

def text_to_speech(text):
    tts = gTTS(text=text, lang='en')
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        return fp.name

# Streamlit app
st.title("Object Detection and Explanation App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Analyze Image'):
        # Detect objects
        detection_prompt = "Identify the main objects in this image."
        detection_result = get_gemini_response(image, detection_prompt)
        
        st.subheader("Detected Objects:")
        st.write(detection_result)
        
        # Get explanation
        explanation_prompt = f"Explain the main object detected in this image: {detection_result}"
        explanation = get_gemini_response(image, explanation_prompt)
        
        st.subheader("Explanation:")
        st.write(explanation)
        
        # Generate audio explanation
        audio_file = text_to_speech(explanation)
        
        # Display audio player
        st.subheader("Audio Explanation:")
        st.audio(audio_file)
        
        # Clean up the temporary audio file
        os.unlink(audio_file)