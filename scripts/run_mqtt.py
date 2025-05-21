from mqtt_client.mqtt import create_mqtt_client, connect_and_loop

if __name__ == "__main__":
    pyClient = create_mqtt_client()
    connect_and_loop(pyClient)