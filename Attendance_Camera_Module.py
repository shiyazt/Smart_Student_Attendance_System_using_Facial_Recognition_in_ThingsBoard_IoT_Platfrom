"""
Date : 06 May 2019
Author : Shiyaz T
Reference : https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/
"""
import pickle
import cv2
import os
import numpy as np
import imutils
import time
import threading
import json
import paho.mqtt.client as mqtt
from collections import Counter
import schedule

#ThingsBoard Platform credentials
# THINGSBOARD_HOST = '192.168.1.7'
THINGSBOARD_HOST = '192.168.43.54'

ACCESS_TOKEN = 'tWCPhSOFvQgM8a50nphG'
INTERVAL = 3
client = mqtt.Client()
client.username_pw_set(ACCESS_TOKEN)
client.connect(THINGSBOARD_HOST, 1883, 60)
client.loop_start()
students = {"shiyaz_t": "absent", "niyaz_t": "absent", "faris": "absent", "mohammed": "absent", "noufal": "absent"}

classStatistics = {"NoPre":0, "NoAbs":0, "Total":0}

class VideoCamera(object):
    def __init__(self):

        # Using OpenCV to capture from device 0. If you have trouble capturing
        # from a webcam, comment the line below out and use a video file
        # instead.
        BASE_DIR = os.path.dirname(__file__)
        print("[INFO] BASE DIR: ", BASE_DIR)

        # self.students = {"shiyaz_t": "absent", "niyaz_t": "absent"}
        # load our serialized face detector from disk
        print("[INFO] loading face detector...")
        protoPath = os.path.join(BASE_DIR, "face_detection_model/deploy.prototxt")
        modelPath = os.path.join(BASE_DIR, "face_detection_model/res10_300x300_ssd_iter_140000.caffemodel")
        self.detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        # load our serialized face embedding model from disk
        embedding_model = os.path.join(BASE_DIR, 'openface_nn4.small2.v1.t7')
        print("[INFO] loading face recognizer...")
        self.embedder = cv2.dnn.readNetFromTorch(embedding_model)

        # load the actual face recognition model along with the label encoder
        recognizer_file = os.path.join(BASE_DIR, 'output/recognizer.pickle')
        le_file = os.path.join(BASE_DIR, 'output/le.pickle')
        self.recognizer = pickle.loads(open(recognizer_file, "rb").read())
        self.le = pickle.loads(open(le_file, "rb").read())

        # Start IoT Platform Thread Process
        t1 = threading.Thread(target=studentData, args=())
        t1.start()
        print("[INFO] starting video stream...")
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):

        ret, frame = self.video.read()
        frame = imutils.resize(frame, width=600)
        (h, w) = frame.shape[:2]

        # construct a blob from the image
        imageBlob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False, crop=False)

        # apply OpenCV's deep learning-based face detector to localize
        # faces in the input image
        self.detector.setInput(imageBlob)
        detections = self.detector.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > 0.99992:
                print("[DEBUG] confidence: ", float(confidence))
                # compute the (x, y)-coordinates of the bounding box for
                # the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # extract the face ROI
                face = frame[startY:endY, startX:endX]
                (fH, fW) = face.shape[:2]

                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue

                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                self.embedder.setInput(faceBlob)
                vec = self.embedder.forward()

                # perform classification to recognize the face
                preds = self.recognizer.predict_proba(vec)[0]
                j = np.argmax(preds)
                proba = preds[j]
                name = self.le.classes_[j]

                # draw the bounding box of the face along with the
                # associated probability
                text = "{}: {:.2f}%".format(name, proba * 100)
                msg = "Detected Person: " + str(name)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              (0, 0, 255), 2)
                cv2.putText(frame, text, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                cv2.putText(frame, msg, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2 )
                print("[DEBUG] Detected Person: ", text)

                if name in students:
                    students[name] = "present"
                    #print(students)
        # We are using Motion JPEG, but OpenCV defaults to capture raw images,
        # so we must encode it into JPEG in order to correctly display the
        # video stream.
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

def resetAttendance():
    resetSwitch = {"reset":0}
    print("[DEBUG] Resetting ....")
    resetSwitch['reset'] = 1
    client.publish('v1/devices/me/telemetry', json.dumps(resetSwitch), 1)
    time.sleep(5)
    resetSwitch['reset'] = 0
    for x in students:
        students[x] = "absent"
    client.publish('v1/devices/me/telemetry', json.dumps(resetSwitch), 1)
    classStatistics['NoPre'] = Counter(students.values())["present"]
    classStatistics['NoAbs'] = Counter(students.values())["absent"]



# Thread FUnction
def studentData():
    print("[DEBUG] Initializing ThingsBoard")
    next_reading = time.time()
    # client = mqtt.Client()
    # client.username_pw_set(ACCESS_TOKEN)
    # client.connect(THINGSBOARD_HOST, 1883, 60)
    # client.loop_start()
    schedule.every(5).minutes.do(resetAttendance)
    classStatistics['Total'] = len(students)
    client.publish('v1/devices/me/telemetry', json.dumps(classStatistics), 1)
    while True:
        client.publish('v1/devices/me/telemetry', json.dumps(students), 1)
        client.publish('v1/devices/me/telemetry', json.dumps(classStatistics), 1)
        # print("[DEBUG] Class Statistics:", str(classStatistics))
        classStatistics['NoPre'] = Counter(students.values())["present"]
        classStatistics['NoAbs'] = Counter(students.values())["absent"]
        next_reading += INTERVAL
        sleep_time = next_reading - time.time()
        schedule.run_pending()
        if sleep_time > 0:
            time.sleep(sleep_time)




