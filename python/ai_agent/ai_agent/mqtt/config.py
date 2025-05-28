import os

MQTT_BROKER = os.getenv("MQTT_BROKER", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "ai-agent")
MQTT_TOPIC = os.getenv("MQTT_PUBLISH_TOPIC", "environment/audio")
