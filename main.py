import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
path += "\\models"

sys.path.append(path)

from add_face import addFace
from face_train import trainFace
from face_reco import faceReco


def run():
    
    print(
    """
    Welcome to Face Recognition 101
    *************************************
    1. Add your face
    2. Face recognition
    3. Train face // will automatically execute
    'q' to exit the program
    """)

    opt = input("Enter your choice : ")
    if opt not in ['1','2','3','q']:
        print("Wrong Choice Restarting Face Recognition")
    elif opt == '1':
        addFace().run()
        trainFace().run()
    elif opt == '2':
        faceReco().run()
    elif opt == '3':
        trainFace().run()
    elif opt == 'q':
         exit()
    run()

if __name__ == "__main__":
    run()
