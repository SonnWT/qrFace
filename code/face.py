import numpy as np
from core.arcface import extract_embedding
from core.faiss_db import get_embedding_by_user_id, cosine_similarity

SIMILARITY_THRESHOLD = 0.55  # sesuaikan kebutuhan keamanan


def verify_face_for_user(image, user_id):
    """
    image: numpy array (BGR)
    user_id: ID dari QR / kartu
    """

    new_emb = extract_embedding(image)
    if new_emb is None:
        return {
            "success": False,
            "message": "Face not detected"
        }

    stored_emb = get_embedding_by_user_id(user_id)
    if stored_emb is None:
        return {
            "success": False,
            "message": "User not registered"
        }

    confidence = cosine_similarity(new_emb, stored_emb)

    if confidence < SIMILARITY_THRESHOLD:
        return {
            "success": False,
            "message": "Face verification failed",
            "confidence": confidence
        }

    return {
        "success": True,
        "user_id": user_id,
        "confidence": confidence
    }
