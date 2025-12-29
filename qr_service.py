import ssl
import paho.mqtt.client as mqtt
import cv2
import numpy as np
from pathlib import Path
from qr.core import decode_qr_from_frame

BROKER = "localhost"
PORT = 8883

SUB_TOPIC = "topic/qr/input"
PUB_TOPIC = "topic/qr/output"

BASE_DIR = Path(__file__).parent
CA_CERT = BASE_DIR / "certs" / "ca.crt"

image_chunks = {}
expected_chunks = 0
current_frame_id = None


def on_connect(client, userdata, flags, reason_code, properties):
    client.subscribe(SUB_TOPIC)
    print("Connected to broker")


def on_message(client, userdata, msg):
    global image_chunks, expected_chunks, current_frame_id

    data = msg.payload
    if len(data) < 3 + 16 + 16:
        return 

    frame_id = data[0]
    index    = data[1]
    total    = data[2]
    sessionId = data[3:3 + 16].decode('utf-8').rstrip('\x00')
    chunk    = data[3 + 16:]

    if current_frame_id != frame_id:
        image_chunks.clear()
        expected_chunks = total
        current_frame_id = frame_id

    if index >= expected_chunks:
        print("[DROP] invalid chunk index")
        image_chunks.clear()
        return

    image_chunks[index] = chunk

    if len(image_chunks) != expected_chunks:
        return

    print("Image Received")
    print(f"Session ID: {sessionId}")

    jpeg_bytes = b''.join(image_chunks[i] for i in range(expected_chunks))

    if not jpeg_bytes.startswith(b'\xff\xd8') or not jpeg_bytes.endswith(b'\xff\xd9'):
        print("[DROP] invalid JPEG")
        image_chunks.clear()
        return

    img = cv2.imdecode(
        np.frombuffer(jpeg_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if img is None or not isinstance(img, np.ndarray):
        print("[DROP] cv2.imdecode failed")
        image_chunks.clear()
        return

    try:
        qr_result = decode_qr_from_frame(img)
        if qr_result:
            print("[QR]", qr_result)
            client.publish(PUB_TOPIC, f"{qr_result}#{sessionId}", qos=1)
    except Exception as e:
        print("[QR ERROR]", e)


    image_chunks.clear()


client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)  

client.tls_set(
    ca_certs=CA_CERT,
    tls_version=ssl.PROTOCOL_TLSv1_2
)

client.tls_insecure_set(False)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
