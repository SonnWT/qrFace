import faiss
import json
import numpy as np
import os

INDEX_PATH = "data/faiss_index.bin"
MAP_PATH = "data/id_map.json"
EMBEDDING_DIM = 512

# =========================
# INIT / LOAD
# =========================
if os.path.exists(INDEX_PATH):
    index = faiss.read_index(INDEX_PATH)
else:
    index = faiss.IndexFlatIP(EMBEDDING_DIM)

if os.path.exists(MAP_PATH):
    id_map = json.load(open(MAP_PATH))
else:
    id_map = {}


# =========================
# FUNCTIONS
# =========================
# bisa dihilangin kalau udah masuk di mqtt
def add_user_embedding(user_id, embedding):
    """
    Tambahkan user baru ke FAISS
    """
    global index, id_map

    embedding = embedding.astype("float32")
    embedding = np.expand_dims(embedding, axis=0)

    idx = index.ntotal
    index.add(embedding)

    id_map[str(idx)] = user_id

    faiss.write_index(index, INDEX_PATH)
    json.dump(id_map, open(MAP_PATH, "w"))

    return idx


def get_embedding_by_user_id(user_id):
    for idx_str, uid in id_map.items():
        if uid == user_id:
            emb = index.reconstruct(int(idx_str))
            return emb / np.linalg.norm(emb)
    return None


def cosine_similarity(a, b):
    return float(np.dot(a, b))
