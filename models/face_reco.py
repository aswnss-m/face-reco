"""
Python Program to identify if there is a face in front of the camera,
This program uses the pretrained haarcascase classifier of the opencv and idenfity front and side faces in the camera
"""
import os
import numpy as np
import cv2
import pickle
import keyboard
class faceReco():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))[:-7]
    camera = cv2.VideoCapture(0)
    front_face_cascade = cv2.CascadeClassifier(str(BASE_DIR)+ '\\cascades\haarcascade_frontalface_alt2.xml') #front face classifier
    side_face_cascade = cv2.CascadeClassifier(str(BASE_DIR)+ '\\cascades\haarcascade_profileface.xml') #side face classifier

    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(str(BASE_DIR)+'\\trainner.yml')
    NAMES = None
    with open(str(BASE_DIR)+"\\label_names.txt","r") as f:
        names = f.read()
        NAMES = names
    def run(self):
        names_derived = []
        for name in self.NAMES.split("\n"):
            names_derived.append(name.split()[0])
        print(names_derived)
        # id = 0 #used for data collection 
        while True:

            #Capture frame by frame
            ret, frame = self.camera.read()

            # coverting the image into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            front_face = self.front_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            side_face = self.side_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)


            if front_face != ():
                for (x, y, w, h) in front_face:
                    roi = gray[y:y+h, x:x+w] #crops into face 
                    id_,conf_ = self.recogniser.predict(roi)
                    if conf_ >= 55:
                        print(names_derived[id_])
                    else:
                        print("Not recognized")
                    #* draws the rectangle in the face , where to show,starting , ending , color , stroke width
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                for (x, y, w, h) in side_face:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('frame',frame)
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                break  # finishing the loop

        self.camera.release()
        cv2.destroyAllWindows()

        # Need to add dataset of other human faces so that the modal can output value 2 for other faces