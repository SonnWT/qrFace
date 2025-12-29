import ssl
from pathlib import Path
import paho.mqtt.client as mqtt

# ===== Config =====
BROKER = "localhost"
PORT = 8883
SUB_TOPIC = "topic/qr/output"

BASE_DIR = Path(__file__).parent
CA_CERT = BASE_DIR / "certs" / "ca.crt"


# ===== Callbacks (API v2) =====
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print("Connected to broker")
        client.subscribe(SUB_TOPIC)
    else:
        print("Connection failed:", reason_code)


def on_message(client, userdata, msg):
    print(f"[{msg.topic}] {msg.payload.decode("utf-8")}")


# ===== Client setup =====
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

client.tls_set(
    ca_certs=str(CA_CERT),
    tls_version=ssl.PROTOCOL_TLSv1_2
)

client.tls_insecure_set(False)

client.on_connect = on_connect
client.on_message = on_message

client.connect(BROKER, PORT, 60)
client.loop_forever()
