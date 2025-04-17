import streamlit as st
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util
import speech_recognition as sr
import pyttsx3
import tempfile
import os
import threading

# Set Streamlit page config FIRST
st.set_page_config(page_title="🩺 HealthBot - Medical Q&A Chatbot", layout="centered", page_icon="💬")

# Text-to-Speech engine initialization
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)

# Global variable to control speaking thread
tts_thread = None

# Stop TTS function
def stop_tts():
    global tts_thread
    if tts_thread and tts_thread.is_alive():
        tts_engine.stop()

# Load precomputed data and embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("models/preprocessed_data.csv")
    embeddings = np.load("models/question_embeddings.npy")
    return df, embeddings

df, question_embeddings = load_data()
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Function to perform semantic search with threshold
CONFIDENCE_THRESHOLD = 0.5

def get_answer(query, df, embeddings):
    start_time = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    best_idx = np.argmax(similarities)
    confidence = similarities[best_idx]
    response_time = round(time.time() - start_time, 2)
    if confidence >= CONFIDENCE_THRESHOLD:
        return df.iloc[best_idx]['clean_answer'], confidence, response_time
    else:
        return "Sorry, I couldn't find a confident answer to your question. Please try rephrasing.", confidence, response_time

# Capture speech input
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙️ Say \"Hey HealthBot\" followed by your question...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio)
            if "hey healthbot" in query.lower():
                query = query.lower().replace("hey healthbot", "").strip()
                st.success(f"You said: {query}")
                return query
            else:
                st.warning("Please say 'Hey HealthBot' to activate.")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError:
            st.error("Speech Recognition service error.")
    return None

# Speak the chatbot's answer (interruptible)
def speak_answer(answer):
    global tts_thread

    def run_tts():
        stop_tts()
        tts_engine.say(answer)
        tts_engine.runAndWait()

    tts_thread = threading.Thread(target=run_tts)
    tts_thread.start()

# Light theme style
st.markdown("""
    <style>
    .stApp {
        background-color: #f8f9fa;
        color: #212529;
    }
    .stTextInput>div>div>input,
    .stTextArea textarea {
        background-color: #ffffff;
        color: #000000;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 General Health Query Chatbot")
st.caption("Ask health-related questions like:\n- What are symptoms of diabetes?\n- Which medicine is good for fever?\n- What is anxiety?\nPlease avoid unrelated or personal diagnostic questions.")

if 'chat' not in st.session_state:
    st.session_state.chat = []

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input("💬 Ask your health question:", key="input")

with col2:
    use_mic = st.button("🎤 Speak")

if use_mic:
    query_from_speech = recognize_speech()
    if query_from_speech:
        user_query = query_from_speech

if user_query:
    stop_tts()  # Stop current speech before answering new one
    with st.spinner("💡 Thinking..."):
        answer, confidence, response_time = get_answer(user_query, df, question_embeddings)
        st.session_state.chat.append((user_query, answer, confidence, response_time))
        speak_answer(answer)

st.subheader("🗨️ Chat History")
chat_history = st.container()

# Display chat history below input
with chat_history:
    for q, a, c, t in reversed(st.session_state.chat):
        st.markdown(f"""
            <div style='display: flex; align-items: flex-start; gap: 10px;'>
                <div style='font-size: 28px;'>🧑‍💻</div>
                <div><strong>You:</strong> {q}<br></div>
            </div>
            <div style='display: flex; align-items: flex-start; gap: 10px;'>
                <div style='font-size: 28px;'>🤖</div>
                <div><strong>HealthBot:</strong> {a}<br><em>Confidence: {c:.2f} | Response Time: {t}s</em></div>
            </div>
            <hr style='border-color: #ccc;'>
        """, unsafe_allow_html=True)
