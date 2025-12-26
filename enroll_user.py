import cv2
import json
import faiss
import numpy as np
import insightface

# =========================
# CONFIG
# =========================
INDEX_PATH = "data/faiss_index.bin"
MAP_PATH = "data/id_map.json"

USER_ID = input("Masukkan user_id untuk enroll: ")

# ========================= 
# LOAD MODEL
# =========================
face_app = insightface.app.FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# =========================
# LOAD FAISS + MAP
# =========================
index = faiss.read_index(INDEX_PATH)

try:
    with open(MAP_PATH) as f:
        id_map = json.load(f)
except FileNotFoundError:
    id_map = {}

# =========================
# ENROLL FUNCTION
# =========================
def enroll_user(user_id, embedding):
    idx = index.ntotal
    index.add(np.array([embedding]).astype("float32"))
    id_map[str(idx)] = user_id

    faiss.write_index(index, INDEX_PATH)
    json.dump(id_map, open(MAP_PATH, "w"), indent=2)

    print(f"[SUCCESS] User {user_id} berhasil didaftarkan pada index {idx}")

# =========================
# CAMERA LOOP
# =========================
cap = cv2.VideoCapture(0)

print("[INFO] Tekan 'S' untuk simpan wajah, 'Q' untuk keluar")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_app.get(frame)

    for face in faces:
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("ENROLL CAMERA", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        if not faces:
            print("[WARN] Wajah tidak terdeteksi")
            continue

        embedding = faces[0].embedding
        embedding = embedding / np.linalg.norm(embedding)

        enroll_user(USER_ID, embedding)
        break

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
