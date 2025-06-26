import os
import json
import yaml
import faiss
import requests
import streamlit as st
import pdfplumber
from gtts import gTTS
from io import BytesIO
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Load Mistral tokenizer once
mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")

def safe_truncate_prompt(prompt: str, max_tokens: int = 8000) -> str:
    tokens = mistral_tokenizer.encode(prompt, truncation=True, max_length=max_tokens)
    return mistral_tokenizer.decode(tokens, skip_special_tokens=True)

# Constants
DOCS_DIR = "docs"
SUPPORTED_EXTENSIONS = [".txt", ".md", ".pdf", ".json", ".yaml", ".yml"]

# Model
embedder = SentenceTransformer("../all-MiniLM-L6-v2")

# ------------------------
# Utility Functions
# ------------------------

def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8") as f:
                return f.read()

        elif ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)

        elif ext == ".json":
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return json.dumps(data, indent=2)

        elif ext in [".yaml", ".yml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return yaml.dump(data, default_flow_style=False)

    except Exception as e:
        st.warning(f"Error reading {filepath}: {e}")
        return ""

    return ""

def chunk_text(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size - overlap)]


#def chunk_text(text: str, chunk_size: int = 300) -> List[str]:  # Reduced size for better granularity
#    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def build_faiss_index(chunks: List[str]):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(len(embeddings[0]))
    index.add(np.array(embeddings).astype("float32"))
    return index, chunks

def search(query: str, index, chunks, k=20):  # Default k increased
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding).astype("float32"), k)
    return [chunks[i] for i in I[0]]



def query_mistral(context: str, question: str) -> str:
    prompt = f"""You are an assistant answering based on the following retrieved document excerpts:

{context}

Answer the question: {question}

If the answer is not found in the context, say "The answer was not found in the provided documents."""
    
    # Truncate to stay within model limits
    prompt = safe_truncate_prompt(prompt, max_tokens=8000)
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False}
    )
    return response.json().get("response", "[No response]")



def generate_audio(text: str) -> BytesIO:
    tts = gTTS(text)
    audio_io = BytesIO()
    tts.write_to_fp(audio_io)
    audio_io.seek(0)
    return audio_io

# ------------------------
# Streamlit Interface
# ------------------------

st.set_page_config(page_title="DevOps Assistant", layout="wide")
st.title("🤖 DevOps Assistant")
st.caption("Reads .txt, .pdf, .md, .yaml, .json files and answers your questions using Mistral.")

mute = st.checkbox("🔇 Mute Voice Output", value=False)

# Load and process files
all_chunks = []
file_count = 0

for file in os.listdir(DOCS_DIR):
    if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
        filepath = os.path.join(DOCS_DIR, file)
        content = extract_text(filepath)
        chunks = chunk_text(content)
        all_chunks.extend(chunks)
        file_count += 1

if file_count == 0:
    st.warning(f"No supported files found in `{DOCS_DIR}`.")
    st.stop()

index, stored_chunks = build_faiss_index(all_chunks)
st.success(f"✅ Loaded and indexed {len(all_chunks)} chunks from {file_count} file(s).")

# Query interface
query = st.text_input("🔍 Ask a question:")

# 🔢 Add slider for number of chunks to retrieve
k = st.slider("🔢 Number of chunks to retrieve", min_value=1, max_value=20, value=6)

if query:
    with st.spinner("Thinking..."):
        top_chunks = search(query, index, stored_chunks, k=k)
        context = "\n\n---\n\n".join(top_chunks)

        # Build prompt for preview
        prompt = f"""You are an assistant answering based on the following retrieved document excerpts:

{context}

Answer the question: {query}

If the answer is not found in the context, say "The answer was not found in the provided documents."""    

        # Truncate the prompt (used by query_mistral)
        safe_prompt = safe_truncate_prompt(prompt, max_tokens=8000)

        # Get model answer
        answer = query_mistral(context, query)

        st.subheader("🧠 Answer")
        st.write(answer)

        if not mute:
            audio_io = generate_audio(answer)
            st.audio(audio_io, format="audio/mp3")

        with st.expander("📄 Retrieved Context"):
            st.code(context)

        with st.expander("📝 Prompt Sent to Mistral"):
            st.code(safe_prompt)
