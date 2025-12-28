import cv2
from core.arcface import extract_embedding
from core.faiss_db import add_user_embedding

print("=== FACE ENROLLMENT ===")
user_id = input("Masukkan user_id: ").strip()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera gagal dibuka")
    exit()

print("📷 Arahkan wajah ke kamera")
print("Tekan [SPACE] untuk capture & enroll")
print("Tekan [Q] untuk batal")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Enroll Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord(" "):  # SPACE
        emb = extract_embedding(frame)
        if emb is None:
            print("❌ Wajah tidak terdeteksi, coba lagi")
            continue

        idx = add_user_embedding(user_id, emb)
        print(f"✅ User '{user_id}' berhasil didaftarkan (index {idx})")
        break

    elif key == ord("q"):
        print("❌ Enrollment dibatalkan")
        break

cap.release()
cv2.destroyAllWindows()
