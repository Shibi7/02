import cv2
import numpy as  np
kernel =  np.ones((5,5),np.uint8)
img = cv2.imread("Resources/img_2.png")


imggrey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret, imgthres = cv2.threshold(imggrey,160,255,cv2.THRESH_BINARY)
imgblr = cv2.GaussianBlur(imggrey,(7,7),0)
imgcny = cv2.Canny(imgblr,100,100)
imgdil = cv2.dilate(imgcny,kernel,iterations=1)
#cv2.imshow("pengui", img)
contours ,heirarchy = cv2.findContours(imgthres,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
imgcpy= img.copy()
cv2.drawContours(imgcpy,contours,-1,(0,255,0),thickness=1)
#cv2.imshow("n", imggrey)
cv2.imshow("asodf",imgcpy)
cv2.imshow("Imagethreshold",imgthres)
#cv2.imshow("penguin", imgblr)
cv2.imshow("penui", imgcny)
#cv2.imshow("t", imgdil)
cv2.waitKey(0)
cv2.imwrite("Imaget",imgthres)