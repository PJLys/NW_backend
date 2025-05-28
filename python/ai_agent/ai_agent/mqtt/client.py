import json
import time
import paho.mqtt.client as mqtt
from .config import *

class MQTTPublisher:
    def __init__(self):
        self.client = mqtt.Client(client_id=MQTT_CLIENT_ID)
        self.client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

    def publish(self, label, confidence, logging=False):
        payload = json.dumps({
            "label": label,
            "score": round(confidence, 3),
            "timestamp": int(time.time())
        })
        if logging:
            print(f"[MQTT] Publishing: {payload}")
        self.client.publish(MQTT_TOPIC, payload)
