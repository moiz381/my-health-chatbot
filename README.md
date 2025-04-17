# my-health-chatbot

# 🩺 HealthBot - General Health Query Chatbot

**HealthBot** is an intelligent chatbot designed to answer general health-related questions using semantic search and sentence embeddings. It provides quick, informative, and conversational responses to questions about symptoms, medications, diseases, and more.

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b?logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)

---

## 🚀 Features

- 🔍 **Semantic Search** using Sentence Transformers
- 🎙️ **Speech Input** with hotword activation: _"Hey HealthBot"_
- 🗣️ **Text-to-Speech** responses
- 💬 **Chat History** with Q&A style layout
- ✨ **Responsive UI** with Light Theme styling
- ⏱️ Displays **response confidence & time**
- 📌 **Example Questions** & user guidance provided

---


## 🧠 How It Works

1. The dataset of health questions is preprocessed and embedded using `sentence-transformers`.
2. A user asks a question via text or speech.
3. The app finds the most semantically similar question in the dataset.
4. The corresponding answer is returned and optionally spoken aloud.

---

## 🗂️ Project Structure

├── streamlit_app.py # Main Streamlit app ├── utils/ │ └── precompute_embeddings.py # Script to preprocess and embed questions ├── models/ │ ├── preprocessed_data.csv # Preprocessed Q&A data │ └── question_embeddings.npy # Saved sentence embeddings ├── data/ │ └── train_data_chatbot.csv # Raw chatbot training data ├── requirements.txt # Python dependencies └── .streamlit/ └── config.toml # (Optional) UI theming config


---

## 🧪 Example Questions to Ask

- "What are symptoms of diabetes?"
- "Which medicine is used for headache?"
- "What is anxiety?"
- "How to treat high blood pressure?"
- "Is fatigue a sign of COVID-19?"

---

## 💻 Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/healthbot-chat.git
   cd healthbot-chat
   
2. (Optional) Create a virtual environment:
   python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run preprocessing script:
   python utils/precompute_embeddings.py

5. Launch the app:
   streamlit run streamlit_app.py

