import insightface
import numpy as np

face_app = insightface.app.FaceAnalysis(name="buffalo_l")   
face_app.prepare(ctx_id=0, det_size=(640, 640))

def extract_embedding(image):
    faces = face_app.get(image)
    if not faces:
        return None
    emb = faces[0].embedding
    return emb / np.linalg.norm(emb)
