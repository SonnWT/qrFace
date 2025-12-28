import faiss
import json
import os

DATA_DIR = "data"
INDEX_PATH = "data/faiss_index.bin"
MAP_PATH = "data/id_map.json"

os.makedirs(DATA_DIR, exist_ok=True)

EMBEDDING_DIM = 512  # InsightFace (buffalo_l) = 512

# Buat index kosong (L2 distance)
index = faiss.IndexFlatL2(EMBEDDING_DIM)

faiss.write_index(index, INDEX_PATH)

# Buat id_map kosong
with open(MAP_PATH, "w") as f:
    json.dump({}, f, indent=2)

print("✅ FAISS index & id_map.json berhasil dibuat")
