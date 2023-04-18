import cv2
from random import randrange
#from img
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img= cv2.imread(r'C:\Users\SHRINATH\AI_study\aienv\Hm.jpg')
grayscaled_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
cv2.imshow('Face Tracker',img)
cv2.waitKey()

print("Code Completed and stored in ")