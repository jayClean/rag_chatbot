# retriever.py
import faiss
import numpy as np
import pickle
import os

# Create the vector_store directory if it doesn't exist
os.makedirs("vector_store", exist_ok=True)

index_path = "vector_store/index.faiss"
meta_path = "vector_store/docs.pkl"

dim = 384  # MiniLM output dimension

if os.path.exists(index_path):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)
else:
    index = faiss.IndexFlatL2(dim)
    metadata = []

def add_to_index(embedding, chunk):
    global index, metadata
    index.add(np.array([embedding], dtype='float32'))
    metadata.append(chunk)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)

def search_index(query_embedding, top_k=3):
    D, Ii = index.search(np.array([query_embedding], dtype='float32'), top_k)
    return [metadata[i] for i in Ii[0] if i < len(metadata)]
