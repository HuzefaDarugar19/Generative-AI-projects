import streamlit as st
import pickle
import faiss
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama

# ===== Config =====
INDEX_PATH = "faiss_index.idx"
PICKLE_PATH = "chunks.pkl"
EMBED_MODEL = "all-MiniLM-L6-v2"

# ===== Load Model & LLM =====
@st.cache_resource
def load_model():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_llm():
    return Ollama(model="tinyllama")

model = load_model()
llm = load_llm()

# ===== FAISS Search =====
def search_faiss(query, top_k=3):
    index = faiss.read_index(INDEX_PATH)
    with open(PICKLE_PATH, "rb") as f:
        chunks = pickle.load(f)

    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, top_k)

    results = [(chunks[i], distances[0][pos]) for pos, i in enumerate(indices[0])]
    return results

# ===== Answer with Context =====
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

# ===== Streamlit UI =====
st.set_page_config(page_title="HEART IS LIFE", page_icon="❤️")
st.title("❤️ HEALTH Q&A Bot with TinyLlama + FAISS")
st.markdown("Ask questions based on your pre-indexed PDFs.")

user_query = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if user_query.strip():
        with st.spinner("Searching and thinking..."):
            answer = answer_with_context(user_query)
        st.subheader("Answer")
        st.write(answer)
    else:
        st.warning("Please enter a question.")
