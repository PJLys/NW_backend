import paho.mqtt.client as mqtt
from . import config, handlers


def create_mqtt_client():

    # Define the MQTT broker address and port
    broker_address = "mqtt.eclipse.org"
    broker_port = 1883

    # Create a new MQTT client instance
    client = mqtt.Client(client_id=config.CLIENT_ID)

    client.on_connect = handlers.on_connect
    client.on_message = handlers.on_message

    return client

def connect_and_loop(client):
    # Connect to the MQTT broker
    client.connect(config.BROKER, config.PORT, 60)

    # Start the loop to process network traffic and dispatch callbacks
    client.loop_forever()