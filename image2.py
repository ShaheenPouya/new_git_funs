import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# ~~~~~~~~~~~~~~~~~~~~~~~~ importing and showing~~~~~~~~~~~~~~~~~~~~~~~~
img = cv.imread('C:\photos\cat.png')
cv.imshow('kitty', img)

# ~~~~~~~~~~~~~~~~~~~~~~~~ rescale ~~~~~~~~~~~~~~~~~~~~~~~~
# def Rescaleframe(frame, scale=0.75):
#     width = int(frame.shape[1] * scale)
#     height = int(frame.shape[0] * scale)
#     dimensions = (width, height)
#
#     return cv.resize (frame,dimensions, interpolation = cv.INTER_AREA)
#
# resized_image = Rescaleframe(img, 0.6)
# cv.imshow('kittykoochik', resized_image)

# ~~~~~~~~~~~~~~~~~~~~~~~~ Drawing Shapes and text ~~~~~~~~~~~~~~~~~~~~~~~~
# creating a blank image
# blank = np.zeros((500,500,3),dtype = 'uint8')
# cv.imshow('k', blank)

# painting with certain color
#blank[200:300, 350:400] = 0,255,0
# cv.imshow('k', blank)

# line
# cv.line(blank,(55,420), (155,420), (250,150,180), thickness = 10)
# cv.imshow('k', blank)

#rectangular
#cv.rectangle(blank,(0,0), (250,250), (120,120,0), thickness = 2)
#cv.rectangle(blank,(0,0), (blank.shape[1]//2,blank.shape[0]//4), (120,120,0), thickness = cv.FILLED)
#cv.imshow('k', blank)

# Circle
# cv.circle(blank,(150,150), 60,(0,150,180), thickness = 1)
# cv.imshow('k', blank)

#text
# cv.putText(blank, 'Shaheen', (400,400), cv.FONT_HERSHEY_TRIPLEX,.50, (250,0,250), 1)
# cv.imshow('k', blank)


# ~~~~~~~~~~~~~~~~~~~~~~CANNYYYYYYYYYY~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# canny = cv.Canny(img, 250,300)
# cv.imshow ("Cancancany", canny)

#~~~~~~~~~~~~~~~~~~~~~~~~Translation~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def translate (img,x,y):
#     transMatrix = np.float32([[1,0,x],[0,1,y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img,transMatrix,dimensions)
# #
# #
# trr = translate(img,50,-100)
# cv.imshow('translate',trr)

#~~~~~~~~~~~~~~~~~~~~~~~~~~split & Merge~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# b,g,r = cv.split(img)
#
# blank = np.zeros(img.shape[:2], dtype = 'uint8')
# blue = cv.merge([b,blank,blank])
# green = cv.merge([blank,g,blank])
# red = cv.merge ([blank,blank,r])
#
# cv.imshow("b",b)    # baraye mesal ino gozashtam!
# cv.imshow("bl",blue)
# cv.imshow("gr", green)
# cv.imshow("rd", red)
#
# # b = translate(b,0,-5)   # ye maskhare bazi e jaleb!
# # g = translate(g,0,5)
#
# merged = cv.merge([b,g,r])
# cv.imshow("merged", merged)

#~~~~~~~~~~~~~~~~~~~~~~~~~~BLURRRRRRRRRRRRRRRR~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# #average
# average = cv.blur(img,(15,15))
# cv.imshow('Average',average)
#
# #Gaussian
# gauss = cv.GaussianBlur(img,(9,9),0)
# cv.imshow('Gauss',gauss)
#
#Median Blur
# median = cv.medianBlur (img,15)
# cv.imshow('Median', median)
#
# #Bilateral
# bilateral =  cv.bilateralFilter(img, 30, 30,30)
# cv.imshow('Bilateral', bilateral)
#
#~~~~~~~~~~~~~~~~~~~~~~~~~~Bitwise~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~Histogram~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#making grayscale
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY )
# cv.imshow ('gray',gray)

#Grayscale histogram
# gray_hist = cv.calcHist ([gray],[0],None, [256],[0,256])
# plt.figure()
# plt.title('grayscale histog')
# plt.xlabel('bins')
# plt.ylabel("# of pixels")
# plt.plot(gray_hist)
# plt.xlim([0,256])
# plt.show()

# #color hist
# colors = ('b','g','r')
# for i,col in enumerate (colors):
#     hist = cv.calcHist([img],[i],None,[250],[0,256])
#     plt.plot(hist,color = col)
#     plt.xlim([0,256])

#ba median moghayese kardam
# hist = cv.calcHist([img],[1],None, [250],[0,256])
# plt.plot (hist,color='b')
# plt.title('img histog')
#
# hist = cv.calcHist([median],[1],None, [250],[0,256])
# plt.plot (hist,color='r')
# plt.title('median histog')
# plt.show()
# #
# # meidian = cv.calcHist ([median],[0],None, [256],[0,256])
# # plt.title('meidian')
# # plt.xlabel('bins')
# # plt.ylabel("# of pixels")
# # plt.plot(meidian)
# # plt.xlim([0,256])
# # plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Tresholding~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# median = cv.medianBlur (img,15)
# gray = cv.cvtColor(median,cv.COLOR_BGR2GRAY)
#simple tresholding
# threshold, thresh = cv.threshold(gray, 150,255,cv.THRESH_BINARY)
# cv.imshow ('THRESH',thresh)
# threshold, thresh_inv = cv.threshold(gray, 150,255,cv.THRESH_BINARY_INV)
# cv.imshow ('THRESHinv',thresh_inv)

#Adaptive Thresholding
# adaptive_thresh =  cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,21,3)
# cv.imshow ('THRESH',adaptive_thresh)
#
# adaptive_thresh_G =  cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,21,3)
# cv.imshow ('THRESH_G',adaptive_thresh_G)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Edge Detection~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
#
# #Sobel
# sobelx = cv.Sobel(gray,cv.CV_64F,1,0)
# sobely = cv.Sobel(gray,cv.CV_64F,1,0)
# combined_sobel = cv.bitwise_or(sobelx,sobely)
#
# cv.imshow('Sobel_x',sobelx)
# cv.imshow('Sobel_y',sobely)
# cv.imshow('Sobel_Comb',combined_sobel)
#
# #canny
# canny = cv.Canny(gray, 150,175)
# canny = ~canny
# cv.imshow('CANNY!!!!!!!!!!!!',canny)




cv.waitKey(0)
