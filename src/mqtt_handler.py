import json
import paho.mqtt.client as mqtt

class MQTTPublisher:
    def __init__(self, broker_address, port=1883):
        self.client = mqtt.Client()
        self.client.connect(broker_address, port, 60)

    def publish_message(self, topic, message):
        self.client.publish(topic, message)

# Example usage
# mqtt_publisher = MQTTPublisher("broker_address")
# mqtt_publisher.publish_message("mqtt/camera", json.dumps({"topic": "mqtt/camera", "description": "numberplate entry/end"}))
