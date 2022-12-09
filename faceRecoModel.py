"""
Python Program to identify if there is a face in front of the camera,
This program uses the pretrained haarcascase classifier of the opencv and idenfity front and side faces in the camera
"""

import numpy as np
import cv2
import pickle

camera = cv2.VideoCapture(0)
front_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_alt2.xml') #front face classifier
side_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_profileface.xml') #side face classifier

recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read('trainner.yml')

# id = 0 #used for data collection 
while True:

    #Capture frame by frame
    ret, frame = camera.read()

    # coverting the image into grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    front_face = front_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    side_face = side_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

    filename = "_image.png" #used for datacollection purpose only

    if front_face != ():
        for (x, y, w, h) in front_face:
            # print(x, y, w, h)
            # print("front_face")
            # Image = frame[y:y+h, x:x+w] #crops into face 
            # cv2.imwrite("hashir_front_"+str(id)+filename, Image)
            # id+=1
            roi = gray[y:y+h, x:x+w] #crops into face 
            id_,conf_ = recogniser.predict(roi)
            print(id_)
            #* draws the rectangle in the face , where to show,starting , ending , color , stroke width
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        for (x, y, w, h) in side_face:
            # print(x, y, w, h)
            # print("side_face")
            # Image = frame[y:y+h,x:x+w] #crops into face
            # cv2.imwrite("hashir_side_"+str(id)+filename, Image)
            # id+=1
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()

# Need to add dataset of other human faces so that the modal can output value 2 for other faces