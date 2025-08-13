# 📄 HEART HEALTH Q&A Bot with TinyLlama + FAISS

An AI-powered Question Answering bot that can read, understand, and answer questions based on **WHO GUIDELINES** — fast and accurate!  
Built with **FAISS** for semantic search, **Sentence Transformers** for embeddings, and **TinyLlama** for local inference.

---

## 🚀 Features
- **PDF ingestion** → Reads and splits your PDF content into searchable chunks.
- **FAISS vector search** → Finds the most relevant text sections for your query.
- **TinyLlama responses** → Generates natural, context-based answers.
- **Streamlit web UI** → Easy-to-use, interactive interface.
- **Local & Private** → No external API calls — all processing happens locally.

---

## 🛠 Tech Stack
- [Streamlit](https://streamlit.io/) – Web app framework
- [FAISS](https://github.com/facebookresearch/faiss) – Vector search
- [Sentence Transformers](https://www.sbert.net/) – Embedding model
- [LangChain](https://www.langchain.com/) – LLM integration
- [Ollama](https://ollama.ai/) – Local TinyLlama inference

---

---

## ⚡ Quick Start

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/HuzefaDarugar19/Generative-AI-projects.git
cd Generative-AI-projects


## 📂 Project Structure
2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Build the FAISS Index (Run Only Once per PDF Update)
bash : python build_index.py

4️⃣ Launch the Streamlit App
bash : streamlit run app.py

📌 Usage
Place your PDFs in the sample_pdfs/ folder (or update the path in build_index.py).

Run build_index.py to process and index them.

Start app.py and open the browser link provided by Streamlit.

Ask questions — the bot will respond only using the PDF content.

🎯 Example
Question:

What are heart disease prevention methods?

Answer:

Eat a balanced diet, exercise regularly, avoid smoking, and manage stress effectively.

🧠 How It Works
Text Extraction → PyPDF2 reads PDF pages.

Chunking → LangChain splits text into overlapping sections.

Embedding → all-MiniLM-L6-v2 converts text into vectors.

Vector Storage → FAISS stores and searches these vectors.

LLM Reasoning → TinyLlama generates answers based on the retrieved chunks.

📜 License
This project is licensed under the MIT License — free to use and modify.

💡 Future Improvements
PDF upload feature directly in Streamlit UI

Support for multiple embedding models

Option to run with other local LLMs via Ollama

Docker support for easy deployment

❤️ Acknowledgements
Sentence Transformers

FAISS

LangChain

Ollama

Streamlit

---








