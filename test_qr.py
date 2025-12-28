import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol


def preprocess(frame):
    """
    Preprocessing agar QR lebih mudah terbaca
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Reduce noise
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # Adaptive threshold (bagus untuk cahaya tidak stabil)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    return thresh


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Camera tidak bisa dibuka")
        return

    print("📷 Camera aktif. Arahkan QR ke kamera...")
    print("Tekan 'q' untuk keluar")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed = preprocess(frame)

        # 🔒 QR ONLY
        qr_codes = decode(processed, symbols=[ZBarSymbol.QRCODE])

        for qr in qr_codes:
            qr_data = qr.data.decode("utf-8")
            x, y, w, h = qr.rect

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(
                frame,
                qr_data,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            print(f"✅ QR Detected: {qr_data}")

        cv2.imshow("QR Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
