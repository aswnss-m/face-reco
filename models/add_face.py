
import cv2
import os


class addFace():
    camera = cv2.VideoCapture(0)
    front_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_frontalface_alt2.xml')  # front face classifier
    side_face_cascade = cv2.CascadeClassifier('cascades\haarcascade_profileface.xml')  # side face classifier
    id = 0

    def run(self):
        name = input("Enter your name : ")
        name = name.strip().replace(" ","_")
        filename = name + ".png"  # used for datacollection purpose only
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(BASE_DIR, 'dataset')
        os.mkdir(os.path.join(image_dir, name))
        final_path = os.path.join(image_dir, name)
        print("Press q to stop")
        while True:
            # Capture frame by frame
            ret, frame = self.camera.read()

            # coverting the image into grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            front_face = self.front_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            side_face = self.side_face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

            if front_face != ():
                for (x, y, w, h) in front_face:
                    Image = frame[y:y+h, x:x+w]  # crops into face
                    nameOfFile = "front_"+str(self.id)+filename
                    cv2.imwrite(os.path.join(final_path, nameOfFile), Image)
                    self.id += 1

                    # * draws the rectangle in the face , where to show,starting , ending , color , stroke width
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                for (x, y, w, h) in front_face:
                    Image = frame[y:y+h, x:x+w]  # crops into face
                    nameOfFile = "front_"+str(self.id)+filename
                    cv2.imwrite(os.path.join(final_path, nameOfFile), Image)
                    self.id += 1

                    # * draws the rectangle in the face , where to show,starting , ending , color , stroke width
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow('frame', frame)
            if keyboard.is_pressed('q'):  # if key 'q' is pressed 
                break  # finishing the loop

        self.camera.release()
        cv2.destroyAllWindows()

# Need to add dataset of other human faces so that the modal can output value 2 for other faces
