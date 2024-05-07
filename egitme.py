import cv2
import os
from PIL import Image
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
casccadePath = "face.xml"
face_cascade = cv2.CascadeClassifier(casccadePath)
path = "yuzverileri"

def get_Images_and_Labels(path):
    image_paths = [os.path.join(path,f) for f in os.listdir(path)]
    images = []
    labels = []

    for image_path in image_paths:
        image_PIL = Image.open(image_path).convert("L")
        image = np.array(image_PIL,"uint8")
        label = int(os.path.split(image_path)[1].split(".")[0].replace("face-",""))
        print(label)
        faces = face_cascade.detectMultiScale(image)
        for (x,y,w,h) in faces:
            images.append(image[y:y+h,x:x+w])
            labels.append(label)
            cv2.imshow("test",image[y:y+h,x:x+w])
            cv2.waitKey(20)
    return images,labels

images,labels = get_Images_and_Labels(path)
cv2.imshow("test",images[0])
cv2.waitKey(1)

recognizer.train(images,np.array(labels))
recognizer.write("Training/trainer.yml")
cv2.destroyAllWindows()
