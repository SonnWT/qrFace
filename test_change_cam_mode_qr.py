import ssl
from pathlib import Path
import cv2
import paho.mqtt.client as mqtt

BROKER = "localhost"
PORT = 8883
TOPIC = "topic/cam/mode"

BASE_DIR = Path(__file__).parent
CA_CERT = BASE_DIR / "certs" / "ca.crt"

def main():
    # MQTT client
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    client.tls_set(
        ca_certs=str(CA_CERT),
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.tls_insecure_set(False)

    client.connect(BROKER, PORT, 60)

    result, mid = client.publish(TOPIC, "qr", qos=1)
    print("Result:", result)

    client.disconnect()

    print("Change camera mode to QR")


if __name__ == "__main__":
    main()
