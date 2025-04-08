# vector_store.py
import faiss
import os

VECTOR_INDEX_PATH = "vector_store/index.faiss"

def load_faiss_index(path=VECTOR_INDEX_PATH):
    if not os.path.exists(path):
        return None
    return faiss.read_index(path)

def is_index_empty(path=VECTOR_INDEX_PATH):
    index = load_faiss_index(path)
    if not index:
        return True
    return index.ntotal == 0

def get_vector_count(path=VECTOR_INDEX_PATH):
    index = load_faiss_index(path)
    if not index:
        return 0
    return index.ntotal
