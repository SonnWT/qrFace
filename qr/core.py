import cv2

def decode_qr_from_frame(frame):
    """
    Decode QR dari frame kamera
    Return: string QR atau None
    """

    detector = cv2.QRCodeDetector()

    data, points, _ = detector.detectAndDecode(frame)

    if data == "":
        return None

    return data


# CARA PAKAI (taruh di file yang berbeda)
# from qr_decoder import decode_qr_from_frame

# frame -> input gambar
# qr_result = decode_qr_from_frame(frame)
# if qr_result:
#     print("QR:", qr_result)
