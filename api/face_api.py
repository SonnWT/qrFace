from fastapi import APIRouter, UploadFile, File, Form
import cv2
import numpy as np
from core.arcface import extract_embedding
from core.faiss_db import get_embedding_by_user_id, cosine_similarity

router = APIRouter(prefix="/face")

@router.post("/verify")
async def verify_face(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    new_embedding = extract_embedding(img)
    if new_embedding is None:
        return {"success": False, "message": "Face not detected"}

    stored_embedding = get_embedding_by_user_id(user_id)
    if stored_embedding is None:
        return {"success": False, "message": "User not registered"}

    confidence = cosine_similarity(new_embedding, stored_embedding)

    return {
        "success": True,
        "user_id": user_id,
        "confidence": confidence,
    }
