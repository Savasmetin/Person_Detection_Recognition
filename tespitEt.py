#-*- coding:utf-8 -*-
import cv2
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("Training/trainer.yml")
cascadePath = "face.xml"
FaceCascade = cv2.CascadeClassifier(cascadePath)
path = "yuzverileri"
cam = cv2.VideoCapture(0)
while True:
    ret,frame = cam.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = FaceCascade.detectMultiScale(gray,scaleFactor=1.2,minSize=(100,100),minNeighbors=5)
    for (x,y,w,h) in faces:
        tahminEdilenKisi,conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,255),thickness=2)
        if tahminEdilenKisi == 1:
            tahminEdilenKisi = "Savas"
        elif tahminEdilenKisi == 2:
            tahminEdilenKisi = "ali Ulvi"
        else:
            tahminEdilenKisi = "Bilinmeyen kişi"
        cv2.putText(frame,tahminEdilenKisi,(x,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255))
        cv2.imshow("ekran",frame)
    if cv2.waitKey(10) & 0xFF == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
