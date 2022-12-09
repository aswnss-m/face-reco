import numpy as np
import cv2
import os
from PIL import Image
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR,'dataset')
front_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_alt2.xml') #front face classifier

recogniser = cv2.face.LBPHFaceRecognizer_create()


y_label = []
x_train = []
label_id = {}
current_id = 0

for root,dirs,files in os.walk(image_dir):
    for file in files:
            if file.endswith(".png"):

                #? path of the image file
                path = os.path.join(root,file)

                #? name of the folder containing image files
                label = os.path.basename(root).lower()
                # print(label,path)
                if label not in label_id:
                    label_id[label] = current_id
                    current_id += 1

                id_ = label_id[label]

                #? cv2 collects image in np array , here pillow converts the image to black and white
                pil_image = Image.open(path).convert('L')

                #? and numpy convert the image to numpy array
                image_array = np.array(pil_image,'uint8')

                faces = front_face_cascade.detectMultiScale(image_array,scaleFactor=1.5, minNeighbors=5)
                for (x,y,w,h) in faces:
                    roi = image_array[y:y+h,x:x+w]
                    x_train.append(roi)
                    y_label.append(id_)

with open("labels.pickle", "wb") as f:
    pickle.dump(label_id, f)

recogniser.train(x_train, np.array(y_label))
recogniser.save("trainner.yml")