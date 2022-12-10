import numpy as np
import cv2
import os
from PIL import Image
import pickle
import keyboard

class trainFace():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))[:-7]
    image_dir = os.path.join(BASE_DIR,'dataset')
    front_face_cascade = cv2.CascadeClassifier(str(BASE_DIR)+"\\cascades\haarcascade_frontalface_alt2.xml") #front face classifier
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    y_label = []
    x_train = []
    label_id = {}
    current_id = 0

    def run(self):
        print("Training Face...")
        for root,dirs,files in os.walk(self.image_dir):
            for file in files:
                    if file.endswith(".png") or file.endswith(".jpg"):

                        #? path of the image file
                        path = os.path.join(root,file)

                        #? name of the folder containing image files
                        label = os.path.basename(root).lower()
                        # print(label,path)
                        if label not in self.label_id:
                            self.label_id[label] = self.current_id
                            self.current_id += 1

                        id_ = self.label_id[label]

                        #? cv2 collects image in np array , here pillow converts the image to black and white
                        pil_image = Image.open(path).convert('L')

                        #? and numpy convert the image to numpy array
                        image_array = np.array(pil_image,'uint8')

                        faces = self.front_face_cascade.detectMultiScale(image_array,scaleFactor=1.5, minNeighbors=5)
                        for (x,y,w,h) in faces:
                            roi = image_array[y:y+h,x:x+w]
                            self.x_train.append(roi)
                            self.y_label.append(id_)

        with open("labels.pickle", "wb") as f:
            pickle.dump(self.label_id, f)
        with open("label_names.txt", "w") as f:
            string = []
            # f.write("\n".join(self.label_id.keys()))
            for i in self.label_id:
                string.append(str(i) + " " + str(self.label_id[i]))
            f.write("\n".join(string))
            
        print("Training Completed")
        # self.recogniser.train(self.x_train, np.array(self.y_label))
        # self.recogniser.save("trainner.yml")