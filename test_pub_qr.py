import ssl
from pathlib import Path
import cv2
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 8883
TOPIC = "topic/qr/input"

BASE_DIR = Path(__file__).parent
IMAGE_PATH = BASE_DIR / "qrsample.png"
CA_CERT = BASE_DIR / "certs" / "ca.crt"

def main():
    img = cv2.imread(IMAGE_PATH)
    if img is None:
        raise RuntimeError("Failed to load image")

    # Encode image to JPEG
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        raise RuntimeError("Failed to encode image")

    payload = buffer.tobytes()

    # MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.tls_set(
        ca_certs=str(CA_CERT),
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.tls_insecure_set(False)

    client.connect(BROKER, PORT, 60)

    client.publish(TOPIC, payload, qos=1)
    client.disconnect()

    print(f"Image published to {TOPIC} ({len(payload)} bytes)")


if __name__ == "__main__":
    main()
