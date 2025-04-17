# utils/precompute_embeddings.py

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer
import re

# Function to clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

# Load dataset
print("🔄 Loading dataset...")
df = pd.read_csv("data/train_data_chatbot.csv")

# Drop missing values
df.dropna(subset=['short_question', 'short_answer'], inplace=True)

# Clean question and answer columns
print("🧹 Cleaning text...")
df['clean_question'] = df['short_question'].astype(str).apply(clean_text)
df['clean_answer'] = df['short_answer'].astype(str).apply(clean_text)

# Load a better transformer model for semantic search
print("🔍 Loading sentence transformer model (this may take a few seconds)...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Encode all cleaned questions into vector embeddings
print("📐 Generating embeddings...")
questions = df['clean_question'].tolist()
embeddings = model.encode(questions, convert_to_numpy=True, show_progress_bar=True)

# Save embeddings and cleaned data
os.makedirs("models", exist_ok=True)
np.save("models/question_embeddings.npy", embeddings)
df[['clean_question', 'clean_answer']].to_csv("models/preprocessed_data.csv", index=False)

print("✅ Preprocessing and embedding complete! Files saved to /models/")
