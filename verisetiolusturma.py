# -*- coding: utf-8 -*-
import cv2

camera = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("face.xml")
kisi_id = input("lütfen idnizi tanımlayınız: ")
i=0
while True:
    i+=1
    ret,frame = camera.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(image=gray,scaleFactor=1.2,minNeighbors=5,
                                      minSize=(100,100))
    for x,y,w,h in faces:
        cv2.imwrite("yuzverileri/face-"+kisi_id+"."+str(i)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(frame,(x,y),(x+w,y+h),thickness=2,color=(255,255,255))
        cv2.imshow("yuz",frame)
        cv2.waitKey(10)
    if i == 50:
        break
camera.release()
cv2.destroyAllWindows()
