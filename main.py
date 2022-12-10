import sys
import os

path = os.path.dirname(os.path.abspath(__file__))
path += "\\models"

sys.path.append(path)






def run():
    
    print(
    """
    Welcome to Face Recognition 101
    *************************************
    1. Add your face
    2. Face recognition
    3. Train face
    'q' to exit the program
    """)

    opt = input("Enter your choice : ")
    if opt not in ['1','2','3','q']:
        print("Wrong Choice Restarting Face Recognition")
    elif opt == '1':
        from add_face import addFace
        addFace().run()
    elif opt == '2':
        from face_reco import faceReco
        faceReco().run()
    elif opt == '3':
        from face_train import trainFace
        trainFace().run()
    elif opt == 'q':
         exit()
    run()

if __name__ == "__main__":
    run()
