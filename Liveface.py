import numpy as np
import cv2

camera = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_alt2.xml')
id = 0
while True:
    #Capture frame by frame
    ret, frame = camera.read()

    # haar classifier works on grayscale images
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6)

    for (x, y, w, h) in faces:
        print(x, y, w, h)
        Image_gray = gray[x:x+w, y:y+h]
        image_filename= 'image_filename ' + str(id)+".png"
        # id+=1
        cv2.imwrite(image_filename, Image_gray)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(20) and 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()