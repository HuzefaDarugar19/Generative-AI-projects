"""
WHO Guidelines ingestion + chunking (HF/LangChain style)
--------------------------------------------------------
- Extract text from PDFs
- Chunk using LangChain's RecursiveCharacterTextSplitter
- Store in Hugging Face Dataset format
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import Dataset
import os
import faiss
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path
from PyPDF2 import PdfReader

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

PDF_DIR = r"C:\Users\win\Desktop\NLP\Gen AI projects"

def process_pdfs():
    """Extract and chunk all PDFs in the specified directory."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    all_chunks = []

    # Loop through all PDF files in the directory
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Processing: {pdf_path}")

            # Extract text from PDF
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            # Split into chunks
            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)

    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks

# Run function
chunks = process_pdfs()
print(chunks[:5])  # Show first 5 chunks