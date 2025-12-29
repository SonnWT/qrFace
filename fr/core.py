import numpy as np
from utils.arcface import extract_embedding
from utils.faiss_db import get_embedding_by_user_id, cosine_similarity


def verify_face_for_user(image, user_id):
    """
    image   : numpy array (BGR)
    user_id : ID dari QR / kartu
    """

    # Extract embedding dari image baru
    new_emb = extract_embedding(image)
    if new_emb is None:
        return f"{False}#"

    # Ambil embedding user dari database
    stored_emb = get_embedding_by_user_id(user_id)
    if stored_emb is None:
        return f"{False}#"

    # Hitung similarity
    confidence = cosine_similarity(new_emb, stored_emb)

    # returnnya bisa disesuaikan dengan keinginan core
    return f"{True}#{float(confidence)}"
