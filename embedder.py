# embedder.py
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')  # 384-dim

def embed_text(text: str):
    return model.encode(text, convert_to_numpy=True).tolist()
