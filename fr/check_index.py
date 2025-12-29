import faiss
import json

index = faiss.read_index("data/faiss_index.bin")
id_map = json.load(open("data/id_map.json"))

print("Total embedding:", index.ntotal)
print("Mapping:", id_map)
