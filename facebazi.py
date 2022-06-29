import cv2 as cv

img = cv.imread('C:\photos/4.jpg')

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
haar_cas = cv.CascadeClassifier('haar_face.xml')
face_rectang = haar_cas.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=8)

median = cv.medianBlur (img,9)
gray_m = cv.cvtColor(median,cv.COLOR_BGR2GRAY)
haar_cas_m = cv.CascadeClassifier('haar_face.xml')
face_rectang_m = haar_cas.detectMultiScale(gray_m,scaleFactor=1.1,minNeighbors=5)


print (f'number of faces found = {len(face_rectang)} vs {len(face_rectang_m)}')

for (x,y,w,h) in face_rectang:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('Chehre', img)

for (x,y,w,h) in face_rectang_m:
    cv.rectangle(median,(x,y),(x+w,y+h),(0,255,0),thickness=2)

cv.imshow('blur shode', median)



cv.waitKey(0)
