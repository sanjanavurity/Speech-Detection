import streamlit as st
import requests
import pandas as pd
import speech_recognition as sr
from PIL import Image
import pytesseract
import io
import os

st.markdown("""
    <style>
        body {
            background: linear-gradient(to right, #dfe9f3, #ffffff, #dfe9f3);
        }
        .stApp {
            background: linear-gradient(to right, #bbdefb, #dff1fe);
        }
        /* Sidebar section customization */
        section[data-testid="stSidebar"] > div:first-child {
            padding: 1rem;
            background: linear-gradient(to right, #bbdefb, #e3f2fd);
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        /* Optional: make Record button more colorful */
        button[kind="secondary"] {
            background-color: #2196f3 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)


# Set the title of the app
st.title('🔍Cyber Hate Detection')

# ✅ Maintain recognized text in session state
if "recognized_text" not in st.session_state:
    st.session_state.recognized_text = ""
if "history_file" not in st.session_state:
    st.session_state.history_file = "search_history.csv"

# 🎤 Sidebar for voice input
st.sidebar.title("🎤 Voice Input")
if st.sidebar.button("Record"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.sidebar.info("Listening...")
        audio = recognizer.listen(source)
    try:
        st.session_state.recognized_text = recognizer.recognize_google(audio)
        st.sidebar.success("✅ Speech recognized!")
    except sr.UnknownValueError:
        st.sidebar.warning("❌ Could not recognize speech!")

st.sidebar.subheader("📷 Image Input")
uploaded_file = st.sidebar.file_uploader("Upload an image or .txt file", type=["png", "jpg", "jpeg", "txt"])
if uploaded_file is not None:
    if uploaded_file.name.endswith(".txt"):
        stringio = io.StringIO(uploaded_file.getvalue().decode("utf-8"))
        txt_content = stringio.read()
        st.session_state.recognized_text = txt_content
        st.sidebar.success("Text extracted from .txt file!")
    else:
        image = Image.open(uploaded_file)
        extracted_text = pytesseract.image_to_string(image)
        st.session_state.recognized_text = extracted_text
        st.sidebar.success("Text extracted from image!")

# Display the recognized text (read-only)
st.text_area("Recognized Text", st.session_state.recognized_text, height=100, disabled=True)

# Input field pre-filled with recognized/processed text
user_input = st.text_area("Enter text to detect:", value=st.session_state.recognized_text, height=150)

# 🚀 Detect Button
if st.button('Detect'):
    if user_input.strip():
        url = 'http://127.0.0.1:5000/predict'
        try:
            response = requests.post(url, json={'text': user_input})
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                st.success(f"✅ Prediction: {prediction}")

                # Emoji Feedback
                if "Non-Hate Speech" in prediction:
                    st.markdown("😃 **Safe Text! No cyberbullying detected.**")
                else:
                    st.markdown("⚠️ **Cyberbullying detected! Please be cautious.**")

                # Append to Search History
                new_row = pd.DataFrame([{"Text": user_input, "Prediction": prediction}])
                if os.path.exists(st.session_state.history_file):
                    history_df = pd.read_csv(st.session_state.history_file)
                    history_df = pd.concat([new_row, history_df], ignore_index=True)
                else:
                    history_df = new_row
                history_df.to_csv(st.session_state.history_file, index=False)
            else:
                st.error("❌ Error: Could not reach the server.")
        except Exception as e:
            st.error(f"⚠️ Exception: {e}")
    else:
        st.warning("⚠️ Please enter or provide some text.")


# 🕘 Display Previous Searches
st.subheader("📜 Previous Searches")
try:
    df = pd.read_csv(st.session_state.history_file)
    st.dataframe(df)

    # 📤 Download history as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download History as CSV",
        data=csv,
        file_name="search_history.csv",
        mime="text/csv"
    )

    # 🔁 Clear history button
    if st.button("🗑️ Clear History"):
        os.remove(st.session_state.history_file)
        st.success("✅ History cleared. Please refresh to see changes.")

except FileNotFoundError:
    st.info("ℹ️ No previous search history available.")

