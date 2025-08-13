import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss

# ----- Config -----
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PDF_DIR = r"C:\Users\win\Desktop\NLP\Gen AI projects"
INDEX_PATH = "faiss_index.bin"
PICKLE_PATH = "chunks.pkl"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ----- Step 1: Process PDFs and create chunks -----
def process_pdfs():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )

    all_chunks = []
    for filename in os.listdir(PDF_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(PDF_DIR, filename)
            print(f"Processing: {pdf_path}")

            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""

            chunks = splitter.split_text(text)
            all_chunks.extend(chunks)

    print(f"✅ Total chunks created: {len(all_chunks)}")
    return all_chunks

# ----- Step 2: Create FAISS index -----
def create_faiss_index(chunks):
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(PICKLE_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"✅ FAISS index saved to {INDEX_PATH}")
    print(f"✅ Chunks saved to {PICKLE_PATH}")
    return model

# ----- Step 3: Search FAISS index -----
def search_faiss(query, model, top_k=3):
    index = faiss.read_index(INDEX_PATH)
    with open(PICKLE_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)
    results = [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]
    return results

# ===== MAIN =====
if __name__ == "__main__":
    chunks = process_pdfs()
    model = create_faiss_index(chunks)

    # Example search
    query = "heart disease prevention"
    results = search_faiss(query, model)
    print("\n🔍 Search Results:")
    for text, score in results:
        print(f"[Score: {score:.4f}] {text[:200]}...\n")
