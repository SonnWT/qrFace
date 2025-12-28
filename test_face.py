import cv2
from core.arcface import extract_embedding
from core.faiss_db import get_embedding_by_user_id, cosine_similarity

SIMILARITY_THRESHOLD = 0.55


def verify_face(frame, user_id):
    new_emb = extract_embedding(frame)
    if new_emb is None:
        return None, "Face not detected"

    stored_emb = get_embedding_by_user_id(user_id)
    if stored_emb is None:
        return None, "User not registered"

    confidence = cosine_similarity(new_emb, stored_emb)

    if confidence < SIMILARITY_THRESHOLD:
        return confidence, "Verification failed"

    return confidence, "VERIFIED"


def main():
    print("=== FACE VERIFICATION CAMERA TEST ===")
    user_id = input("Masukkan user_id (dari QR): ").strip()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera tidak bisa dibuka")
        return

    print("📷 Arahkan wajah ke kamera")
    print("Tekan Q untuk keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        confidence, status = verify_face(frame, user_id)

        if confidence is not None:
            label = f"{status} ({confidence:.2f})"
            color = (0, 255, 0) if status == "VERIFIED" else (0, 0, 255)

            cv2.putText(
                frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
            )

            print(label)

        cv2.imshow("Face Verification Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
