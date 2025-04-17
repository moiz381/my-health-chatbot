import streamlit as st
import pandas as pd
import numpy as np
import time
from sentence_transformers import SentenceTransformer, util

# Set Streamlit page config FIRST
st.set_page_config(page_title="🩺 HealthBot - Medical Q&A Chatbot", layout="centered", page_icon="💬")

# Load precomputed data and embeddings
@st.cache_data
def load_data():
    df = pd.read_csv("models/preprocessed_data.csv")
    embeddings = np.load("models/question_embeddings.npy")
    return df, embeddings

df, question_embeddings = load_data()
model = SentenceTransformer("all-MiniLM-L6-v2")

# Function to perform semantic search
def get_answer(query, df, embeddings, threshold=0.4):
    start_time = time.time()
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, embeddings)[0].cpu().numpy()
    best_idx = np.argmax(similarities)
    confidence = similarities[best_idx]
    response_time = round(time.time() - start_time, 2)
    if confidence < threshold:
        return "I'm not sure how to answer that. Please try rephrasing your question.", confidence, response_time
    return df.iloc[best_idx]['clean_answer'], confidence, response_time

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
st.caption("Ask health-related questions about symptoms, diseases, medications, mental health, and more.\nExample: 'What are symptoms of diabetes?', 'Which medicine is good for fever?', 'What is anxiety?'")

if 'chat' not in st.session_state:
    st.session_state.chat = []

col1, col2 = st.columns([3, 1])

with col1:
    user_query = st.text_input("💬 Ask your health question:", key="input")

if user_query:
    with st.spinner("💡 Thinking..."):
        answer, confidence, response_time = get_answer(user_query, df, question_embeddings)
        st.session_state.chat.append((user_query, answer, confidence, response_time))

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
