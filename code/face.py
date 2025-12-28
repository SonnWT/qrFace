import numpy as np
from core.arcface import extract_embedding
from core.faiss_db import get_embedding_by_user_id, cosine_similarity


def verify_face_for_user(image, user_id):
    """
    image   : numpy array (BGR)
    user_id : ID dari QR / kartu
    """

    # Extract embedding dari image baru
    new_emb = extract_embedding(image)
    if new_emb is None:
        return {
            "success": False,
            "message": "Face not detected"
        }

    # Ambil embedding user dari database
    stored_emb = get_embedding_by_user_id(user_id)
    if stored_emb is None:
        return {
            "success": False,
            "message": "User not registered"
        }

    # Hitung similarity
    confidence = cosine_similarity(new_emb, stored_emb)

    # returnnya bisa disesuaikan dengan keinginan core
    return {
        "success": True,
        "user_id": user_id,
        "confidence": float(confidence)
    }
