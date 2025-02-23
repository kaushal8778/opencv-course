#pylint:disable=no-member
# Haarcascades and binary classifier are usedd for fact detection. Binary classifier is more advanced. 

import cv2 as cv

img = cv.imread('F:\KAUSHAL\Internship\OpenCV\opencv-course\Resources\Photos\group 1.jpg')
cv.imshow('Group of 5 people', img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray People', gray)

haar_cascade = cv.CascadeClassifier('F:\KAUSHAL\Internship\OpenCV\opencv-course\Section #3 - Faces\haar_face.xml')

faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1)

print(f'Number of faces found = {len(faces_rect)}')

for (x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0), thickness=2)

cv.imshow('Detected Faces', img)



cv.waitKey(0)