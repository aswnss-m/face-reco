# Face Recognition using open cv 
This repository contains an open cv model that can be used to trian and recognize your face. The model uses haarcascade method to find the face and using LBPHFaceRecognizer it checks the similarity between the dataset and the face.

## filesystem structure
 *root <br>
|- cascade // contains the haarcascade model <br>
|- dataset // contains the dataset *which is not tracked by git <br>
|- add_face.py // Used to add a new face in to the dataset <br>
|- face_train.py // Used to train the model for faces in the     dataset <br>
|- faceRecoModel.py // The face recognition model<br>


