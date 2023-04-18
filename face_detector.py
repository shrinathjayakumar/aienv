import cv2
from random import randrange

# face tracking trained in this xml ,as options for training different 
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# reading image 

# img= cv2.imread(r'C:\Users\SHRINATH\AI_study\aienv\HS.jpg')
img= cv2.imread(r'C:\Users\SHRINATH\AI_study\aienv\Hm.jpg')

# changing color scale to gray
grayscaled_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# detect face co-ordinates
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
print(face_coordinates)
# cv2.rectangle(img,(72,31),(72+51,31+51),(0,255,0),10)
# mapping them to x,y,w,h
for (x,y,w,h) in face_coordinates:
# (x,y,w,h)= face_coordinates[0]
# applying equation
    cv2.rectangle(img,(x,y),(x+w,y+h),(randrange(256),randrange(256),randrange(256)),2)
# showing image current situation
cv2.imshow('Face Tracker',img)
# holding till next action
cv2.waitKey()

print("Code Completed and stored in ")