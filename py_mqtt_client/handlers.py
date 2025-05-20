def on_connect(client, userdata, flags, rc):
    """Callback function for when the client connects to the broker."""
    print(f"Connected with result code {rc}")
    # Subscribe to a topic after connecting
    client.subscribe("test/topic")
    print("Subscribed to topic: test/topic")

def on_message(client, userdata, msg):
    """Callback function for when a message is received from the broker."""
    print(f"Message received: {msg.topic} {msg.payload.decode()}")
