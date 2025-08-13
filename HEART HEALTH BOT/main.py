import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# ----- Config -----
INDEX_PATH = "faiss_index.idx"
PICKLE_PATH = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# Load model (must be same as in create_index.py)
model = SentenceTransformer(EMBED_MODEL)
llm = Ollama(model="tinyllama")

# ----- Search FAISS -----
def search_faiss(query, top_k=3):
    index = faiss.read_index(INDEX_PATH)
    with open(PICKLE_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]
    return results

# ----- Answer with Context -----
def answer_with_context(query):
    results = search_faiss(query)
    context = "\n".join([text for text, score in results])

    prompt = f"""
You are an assistant that answers questions using ONLY the information in the context.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question:
{query}

Answer:
"""
    return llm(prompt)

# ===== MAIN =====
if __name__ == "__main__":
    query = "What are heart disease prevention methods?"
    print(answer_with_context(query))
