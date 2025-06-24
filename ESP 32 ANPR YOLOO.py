# mqtt_ocr_door_control.py
import cv2
import pandas as pd
import easyocr
import json
import os
from datetime import datetime
import re
import time
import paho.mqtt.client as mqtt


class MQTTOCRController:
    def __init__(self):
        """Initialize MQTT OCR system"""
        # Initialize EasyOCR
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        print("OCR initialized")

        # File paths
        self.reservations_file = r"D:\anpr\parking_data\reservations_history.csv"

        # MQTT settings
        self.mqtt_broker = "broker.hivemq.com"  # Free public broker
        self.mqtt_port = 1883
        self.mqtt_topic_entry = "parking/door/entry"
        self.mqtt_topic_exit = "parking/door/exit"
        self.mqtt_client_id = "PC_OCR_Controller"

        # Setup MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.on_disconnect = self.on_mqtt_disconnect

        # Connect to MQTT broker
        self.connect_mqtt()

        # Access control variables
        self.last_access_granted = False
        self.access_timeout = 5  # seconds
        self.last_detection_time = 0

    def connect_mqtt(self):
        """Connect to MQTT broker"""
        try:
            print(f"Connecting to MQTT broker: {self.mqtt_broker}")
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port, 60)
            self.mqtt_client.loop_start()
            print("✓ MQTT client started")
        except Exception as e:
            print(f"MQTT connection failed: {e}")

    def on_mqtt_connect(self, client, userdata, flags, rc):
        """Callback for MQTT connection"""
        if rc == 0:
            print("✓ Connected to MQTT broker successfully!")
            # Subscribe to exit door notifications
            client.subscribe(self.mqtt_topic_exit)
            print(f"Subscribed to: {self.mqtt_topic_exit}")

            # Send connection status
            status_msg = {
                "status": "PC_OCR_CONNECTED",
                "timestamp": datetime.now().isoformat(),
                "system": "ready"
            }
            client.publish("parking/status", json.dumps(status_msg))
        else:
            print(f"MQTT connection failed with code {rc}")

    def on_mqtt_message(self, client, userdata, msg):
        """Handle incoming MQTT messages"""
        try:
            topic = msg.topic
            message = json.loads(msg.payload.decode())

            if topic == self.mqtt_topic_exit:
                print(f"Exit door notification: {message}")
        except Exception as e:
            print(f"Error processing MQTT message: {e}")

    def on_mqtt_disconnect(self, client, userdata, rc):
        """Handle MQTT disconnection"""
        print("MQTT client disconnected")

    def send_door_command(self, access_granted, detected_plate=""):
        """Send door command via MQTT"""
        message = {
            "access_granted": access_granted,
            "detected_plate": detected_plate,
            "timestamp": datetime.now().isoformat(),
            "status": "OPEN" if access_granted else "CLOSED",
            "system": "OCR_CONTROL"
        }

        try:
            result = self.mqtt_client.publish(self.mqtt_topic_entry, json.dumps(message))
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"✓ MQTT message sent: {message['status']} - Plate: {detected_plate}")
            else:
                print(f"✗ MQTT publish failed: {result.rc}")
        except Exception as e:
            print(f"Error sending MQTT message: {e}")

    def _clean_text(self, text):
        """Clean OCR text"""
        return re.sub(r'[^A-Z0-9]', '', text.upper())

    def _get_expected_plates(self):
        """Get expected license plates from CSV"""
        try:
            df = pd.read_csv(self.reservations_file)
            plates = []
            for plate in df['plate_number'].values:
                clean_plate = self._clean_text(str(plate))
                if clean_plate and len(clean_plate) >= 4:
                    plates.append(clean_plate)
            return plates
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return []

    def read_text_from_image(self, image):
        """Read text from image using OCR"""
        try:
            results = self.ocr_reader.readtext(image)
            detected_texts = []

            for (bbox, text, confidence) in results:
                if confidence > 0.6:
                    clean_text = self._clean_text(text)
                    if len(clean_text) >= 4:
                        detected_texts.append(clean_text)

            return detected_texts
        except Exception as e:
            print(f"OCR error: {e}")
            return []

    def check_access(self, detected_texts):
        """Check if any detected text matches expected plates"""
        expected_plates = self._get_expected_plates()

        print(f"=== ACCESS CHECK ===")
        print(f"Detected: {detected_texts}")
        print(f"Expected: {expected_plates}")

        # Check for exact matches
        for detected in detected_texts:
            for expected in expected_plates:
                if detected == expected:
                    print(f"*** EXACT MATCH: {detected} ***")
                    return True, detected

        # Check for partial matches
        for detected in detected_texts:
            for expected in expected_plates:
                if (len(detected) >= 5 and detected in expected) or \
                        (len(expected) >= 5 and expected in detected):
                    print(f"*** PARTIAL MATCH: {detected} ≈ {expected} ***")
                    return True, detected

        print("*** NO MATCH - ACCESS DENIED ***")
        return False, ""

    def process_camera(self, camera_id=0):
        """Process camera feed"""
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print("Cannot open camera")
            return

        print("=== MQTT CAMERA MONITORING STARTED ===")
        print("Entry Door: MQTT controlled")
        print("Exit Door: ESP32 ultrasonic sensor")
        print("Press 'q' to quit")
        print("======================================")

        frame_count = 0
        no_detection_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Process every 30th frame
            if frame_count % 30 == 0:
                detected_texts = self.read_text_from_image(frame)

                if detected_texts:
                    no_detection_count = 0
                    access_granted, matched_plate = self.check_access(detected_texts)

                    if access_granted:
                        self.last_access_granted = True
                        self.last_detection_time = time.time()

                        # Send MQTT command to ESP32
                        self.send_door_command(True, matched_plate)

                        cv2.putText(frame, f"ACCESS GRANTED: {matched_plate}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, "MQTT: DOOR OPENING", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, "SCANNING...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                else:
                    no_detection_count += 1
                    cv2.putText(frame, "NO PLATE DETECTED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Check timeout
                if self.last_access_granted and \
                        (time.time() - self.last_detection_time > self.access_timeout):
                    self.last_access_granted = False
                    self.send_door_command(False, "")
                    print("Access timeout - door access revoked")

                # No detection for a while
                if no_detection_count > 3:
                    if self.last_access_granted:
                        self.last_access_granted = False
                        self.send_door_command(False, "")

            # Add MQTT status to frame
            connection_status = "CONNECTED" if self.mqtt_client.is_connected() else "DISCONNECTED"
            cv2.putText(frame, f"MQTT: {connection_status}", (10, frame.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0) if connection_status == "CONNECTED" else (0, 0, 255),
                        1)
            cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('MQTT OCR Door Control', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        # Send final close command
        self.send_door_command(False, "")
        print("Camera stopped - door closed")

    def process_image(self, image_path):
        """Process single image for testing"""
        try:
            image = cv2.imread(image_path)
            if image is None:
                print("Cannot load image")
                return

            detected_texts = self.read_text_from_image(image)
            print(f"Detected: {detected_texts}")

            if detected_texts:
                access_granted, matched_plate = self.check_access(detected_texts)
                self.send_door_command(access_granted, matched_plate)

                if access_granted:
                    print(f"✓ ACCESS GRANTED - {matched_plate}")
                else:
                    print("✗ ACCESS DENIED")
            else:
                print("No text detected")
                self.send_door_command(False, "")

        except Exception as e:
            print(f"Error: {e}")

    def cleanup(self):
        """Cleanup MQTT connection"""
        try:
            self.send_door_command(False, "")  # Close door
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("MQTT client disconnected")
        except:
            pass


def main():
    print("=== MQTT DUAL DOOR CONTROL SYSTEM ===")
    print("Entry Door: MQTT + OCR Recognition")
    print("Exit Door: ESP32 Ultrasonic Sensor")
    print("===================================")

    ocr = MQTTOCRController()

    try:
        print("\nSelect mode:")
        print("1. Start camera monitoring")
        print("2. Test single image")
        print("3. Exit")

        while True:
            choice = input("\nChoice: ").strip()

            if choice == '1':
                print("Starting MQTT camera monitoring...")
                ocr.process_camera()
            elif choice == '2':
                image_path = input("Image path: ").strip()
                if os.path.exists(image_path):
                    ocr.process_image(image_path)
                else:
                    print("File not found")
            elif choice == '3':
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        ocr.cleanup()


if __name__ == "__main__":
    main()