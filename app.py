from fastapi import FastAPI
from api.qr_api import router as qr_router
from api.face_api import router as face_router

app = FastAPI(title="QR + Face Recognition AI Server")

app.include_router(qr_router)
app.include_router(face_router)
