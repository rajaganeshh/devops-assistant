🤖 DevOps Assistant

A smart, local AI-powered assistant that reads `.txt`, `.pdf`, `.md`, `.yaml`, and `.json` files and answers your questions using the **Mistral-7B Instruct** model. The assistant leverages **FAISS** for fast document retrieval and provides **audio output** using Google Text-to-Speech (gTTS).

---

## 🚀 Features

- ✅ Supports multiple file types: `.txt`, `.md`, `.pdf`, `.json`, `.yaml`
- 🧠 Uses `sentence-transformers` for semantic embedding
- 🔍 Fast vector search with **FAISS**
- 🤖 Local inference with **Mistral-7B-Instruct**
- 🔊 Optional voice output (Google TTS)
- 🌐 Clean UI with **Streamlit**

---

## 🛠️ Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/devops-assistant.git
cd devops-assistant


2. Create Virtual Environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies

pip install -r requirements.txt
Make sure you have ffmpeg installed if you want to use gTTS audio playback.

4. Place Documents
Place your .txt, .pdf, .md, .json, .yaml files in the docs/ directory.

5. Run Local Mistral Server
Ensure the Mistral 7B model is running locally on: http://localhost:11434/api/generate

Use something like Ollama(https://ollama.com/) or your own Mistral setup.

6. Launch the App
streamlit run streamlit_rag_voice_enabled.py


💡 How It Works
   - Reads all supported files from the docs/ directory.
   - Extracts and splits content into manageable text chunks.
   - Embeds each chunk with all-MiniLM-L6-v2.
   - Builds a FAISS vector index for fast semantic search.
   - Accepts natural language questions from the user.
   - Retrieves relevant chunks and sends them to the Mistral model.
   - Displays the answer and plays audio (optional).

If any queries please send mail to rajaganeshh@gmail.com
