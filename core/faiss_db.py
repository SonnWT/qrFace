import faiss
import json
import numpy as np

INDEX_PATH = "data/faiss_index.bin"
MAP_PATH = "data/id_map.json"

index = faiss.read_index(INDEX_PATH)
id_map = json.load(open(MAP_PATH))

def get_embedding_by_user_id(user_id):
    for idx, uid in id_map.items():
        if uid == user_id:
            return index.reconstruct(int(idx))
    return None

def cosine_similarity(a, b):
    return float(np.dot(a, b))
