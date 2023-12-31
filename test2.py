import cv2
import numpy as np
import pickle
import pandas as pd
from ultralytics import YOLO
import cvzone
import paho.mqtt.publish as publish
import threading
#import time  

# MQTT Configuration
MQTT_BROKER = "dev.coppercloud.in"  # Replace with your MQTT broker address
MQTT_PORT = 1883
MQTT_TOPIC = "space"

with open("freedomtect", "rb") as f:
    data = pickle.load(f)
    polylines, area_name = data['polylines'], data['area_name']

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

model = YOLO('yolov8s.pt')

# Adjust the frame rate as needed
cap = cv2.VideoCapture('easy1.mp4', cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_FPS, 30)
count = 0

def publish_mqtt_message(free_space):
    try:
        if free_space == 0:
            publish.single(MQTT_TOPIC, payload="off", hostname=MQTT_BROKER, port=MQTT_PORT)
        else:
            publish.single(MQTT_TOPIC, payload="on", hostname=MQTT_BROKER, port=MQTT_PORT)
    except Exception as e:
        print(f"Error publishing MQTT message: {e}")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))
    frame_copy = frame.copy()
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []
    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])

        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        if 'car' in c:
            list1.append([cx, cy])

    counter1 = []
    list2 = []
    for i, polyline in enumerate(polylines):
        list2.append(i)
        cv2.polylines(frame, [polyline], True, (0, 255, 0), 2)
        cvzone.putTextRect(frame, f'{area_name[i]}', tuple(polyline[0]), 1, 1)
        for i1 in list1:
            cx1 = i1[0]
            cy1 = i1[1]
            result = cv2.pointPolygonTest(polyline, (cx1, cy1), False)
            if result >= 0:
                cv2.circle(frame, (cx1, cy1), 5, (255, 0, 0), -1)
                cv2.polylines(frame, [polyline], True, (0, 0, 255), 2)
                counter1.append(cx1)

    car_count = len(counter1)
    free_space = len(list2) - car_count
    cvzone.putTextRect(frame, f'carcounter:-{car_count}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'free_space:-{free_space}', (50, 160), 2, 2)

    # Publish parking status to MQTT topic in a separate thread
    mqtt_thread = threading.Thread(target=publish_mqtt_message, args=(free_space,))
    mqtt_thread.start()
    #time.sleep(1)

    cv2.imshow('FRAME', frame)
    key = cv2.waitKey(1) & 0xFF

cap.release()
cv2.destroyAllWindows()
