# The steps implemented in the object detection sample code: 
# 1. for an image of width and height being (w, h) pixels, resize image to (w', h'), where w/h = w'/h' and w' x h' = 262144
# 2. resize network input size to (w', h')
# 3. pass the image to network and do inference
# (4. if inference speed is too slow for you, try to make w' x h' smaller, which is defined with DEFAULT_INPUT_SIZE (in object_detection.py or ObjectDetection.cs))
"""Sample prediction script for TensorFlow 1.x."""
import sys
from dotenv import load_dotenv
import os
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image
from object_detection import ObjectDetection
from mqtt_handler import MQTTPublisher
import pytesseract
import time

MODEL_FILENAME = 'model.pb'
LABELS_FILENAME = 'labels.txt'

tf.compat.v1.disable_eager_execution()

load_dotenv('../.env')

class TFObjectDetection(ObjectDetection):
    """Object Detection class for TensorFlow"""

    def __init__(self, graph_def, labels):
        super(TFObjectDetection, self).__init__(labels)
        self.graph = tf.compat.v1.Graph()
        with self.graph.as_default():
            input_data = tf.compat.v1.placeholder(tf.float32, [1, None, None, 3], name='Placeholder')
            tf.import_graph_def(graph_def, input_map={"Placeholder:0": input_data}, name="")

    def predict(self, preprocessed_image):
        inputs = np.array(preprocessed_image, dtype=np.float64)[:, :, (2, 1, 0)]  # RGB -> BGR

        with tf.compat.v1.Session(graph=self.graph) as sess:
            output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
            outputs = sess.run(output_tensor, {'Placeholder:0': inputs[np.newaxis, ...]})
            return outputs[0]


def perform_ocr(image, bounding_box):
    # Adjust the bounding box coordinates as necessary
    left = int(bounding_box['left'] * image.width)
    top = int(bounding_box['top'] * image.height)
    width = int(bounding_box['width'] * image.width)
    height = int(bounding_box['height'] * image.height)

    # Crop and process the image for OCR
    roi = image.crop((left, top, left + width, top + height))
    text = pytesseract.image_to_string(roi, config='--psm 6')
    return text

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description='ANPR System')
    parser.add_argument('--camera', help='Select the camera package (cv2 or rpi)', default='cv2')
    args = parser.parse_args()

    # Initialize MQTT Publisher
    print("Initializing MQTT Publisher...")
    mqtt_publisher = MQTTPublisher(os.getenv('BROKER_ADDRESS'), int(os.getenv('BROKER_PORT', 1883)))

    # Load TensorFlow model and labels
    print("Loading model...")
    graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(MODEL_FILENAME, 'rb') as f:
        graph_def.ParseFromString(f.read())

    print("Loading labels...")
    with open(LABELS_FILENAME, 'r') as f:
        labels = [label.strip() for label in f.readlines()]

    od_model = TFObjectDetection(graph_def, labels)

    if args.camera == 'rpi':
        from picamera.array import PiRGBArray
        from picamera import PiCamera

        # Using PiCamera
        print("Using PiCamera...")
        camera = PiCamera()
        camera.resolution = (1280, 720)
        camera.framerate = 24
        rawCapture = PiRGBArray(camera, size=(1280, 720))
        time.sleep(0.1)

        try:
            for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
                image = frame.array
                image_pil = Image.fromarray(image)

                # Object detection
                predictions = od_model.predict_image(image_pil)

                # Number plate detection and OCR
                for pred in predictions:
                    if pred['probability'] > 0.8 and pred['tagName'] == 'number plate':
                        print("Number plate detected. Performing OCR...")
                        text = perform_ocr(image_pil, pred['boundingBox'])
                        print("Detected Text:", text)

                        # MQTT Message
                        json_data = {
                            "topic": os.getenv('MQTT_TOPIC'),
                            "description": text + " " + os.getenv('CAMERA_POSITION')
                        }
                        mqtt_publisher.publish_message(os.getenv('MQTT_TOPIC'), json.dumps(json_data))

                # Display the frame
                cv2.imshow("Video Stream", image)
                rawCapture.truncate(0)

        except KeyboardInterrupt:
            print("Stopping video stream...")
        finally:
            camera.close()

    else:
        # Using cv2
        print("Using cv2 for video capture...")
        cap = cv2.VideoCapture('/dev/video0')
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Unable to capture video")
                    break

                # Convert frame to PIL Image
                image_pil = Image.fromarray(frame)

                # Object detection
                predictions = od_model.predict_image(image_pil)

                # Number plate detection and OCR
                for pred in predictions:
                    if pred['probability'] > 0.8 and pred['tagName'] == 'number plate':
                        print("Number plate detected. Performing OCR...")
                        text = perform_ocr(image_pil, pred['boundingBox'])
                        print("Detected Text:", text)

                        # MQTT Message
                        json_data = {
                            "topic": os.getenv('MQTT_TOPIC'),
                            "description": text + " " + os.getenv('CAMERA_POSITION')
                        }
                        mqtt_publisher.publish_message(os.getenv('MQTT_TOPIC'), json.dumps(json_data))

                # Display the current frame in the "Video Stream" window
                cv2.imshow("Video Stream", frame)

        except KeyboardInterrupt:
            print("Stopping video stream...")
        finally:
            cap.release()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

