import cv2
from pyzbar.pyzbar import decode

def decode_qr_from_frame(frame):
    """
    Decode QR dari frame kamera
    Return: string QR atau None
    """

    # ubah jadi gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # gaussian blue (reduce noise)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    # decode qr
    qr_codes = decode(blur)

    if not qr_codes:
        return None

    return qr_codes[0].data.decode("utf-8")


# CARA PAKAI (taruh di file yang berbeda)
# from qr_decoder import decode_qr_from_frame

# frame -> input gambar
# qr_result = decode_qr_from_frame(frame)
# if qr_result:
#     print("QR:", qr_result)
