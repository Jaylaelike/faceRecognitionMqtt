import cv2
import json
import time
import datetime as dt
import paho.mqtt.client as mqtt


detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

broker = "35.247.160.179"
port = 1883
username = 'jay'
password = 'satjay1994'


def on_connect(client, userdata, flags, rc):
    print("Connected with code: ", rc)
    client.subscribe('test/#')


def on_message(client, userdata, msg):
    print(str(msg.payload))


client = mqtt.Client("Python1")
client.on_connect = on_connect
client.on_message = on_message

client.username_pw_set(username, password)
client.connect(broker, port, 60)

client.loop_start()
time.sleep(1)


def facerecognizer():

    recognizer.read("trainerdir/training.yml")

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Loading data

    with open('data.json', 'r') as f:
        names = json.load(f)

    # reverse the data
    # NOTE: for k, v !!
    # else it raises error !
    names = {v: k for k, v in names.items()}
    # print(names)
    print("[INFO] Face recognition is starting..")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    try:

        while True:

            ret, img = cap.read()

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = detector.detectMultiScale(gray,
                                              scaleFactor=1.3,
                                              minNeighbors=5
                                              # minSize = (20,20)
                                              )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

                roi_gray = gray[y:y + h, x:x + w]
                roi_color = img[y:y + h, x:x + w]

                ID, confidence = recognizer.predict(roi_gray)

                if (confidence < 100):

                    ID = names[ID]

                    confidence = "{}%".format(round(100 - confidence))

                    while True:
                        client.publish("Tutorial/", ID + " " + str(dt.datetime.now()))
                        print('ID sent')
                        time.sleep(3)

                    else:
                        client.publish("Tutorial/", 'None')
                        print('ID sent')
                        time.sleep(3)

                    client.loop_forever()

                else:
                    ID = "Unkown"
                    confidence = "{}%".format(round(100 - confidence))
                    print(ID, dt.datetime.now())
                    while True:
                        client.publish("Tutorial/", ID + " " + str(dt.datetime.now()))
                        print('ID sent')
                        time.sleep(3)
                    else:
                        client.publish("Tutorial/", 'None')
                        print('ID sent')
                        time.sleep(3)

                    client.loop_forever()

    # except UnboundLocalError:
    #     print("Error occured. Exitting..")

    except KeyboardInterrupt:
        pass
    except KeyError as K:
        print(K)
        print('[INFO] Name Value is a string and not an integer')

    print("[INFO] Exiting program..")
    cap.release()