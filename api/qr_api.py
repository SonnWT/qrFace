from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
from pyzbar.pyzbar import decode

router = APIRouter(prefix="/qr")

@router.post("/decode")
async def decode_qr(image: UploadFile = File(...)):
    data = await image.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

    qr_codes = decode(img)
    if not qr_codes:
        return {"success": False, "message": "QR not detected"}

    qr_string = qr_codes[0].data.decode("utf-8")

    return {
        "success": True,
        "qr_string": qr_string
    }
