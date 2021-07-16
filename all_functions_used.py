# Python program FOR Hindi CHARACTER RECOGNITION MOSAIC PS1
#TEAM ULTRONIX 
#TEAM LEADER-RISHAB ARYA

#our code is invariant to rotation,
#the word can be anywhere not necessary in middle,
#works on blurry images,
#works on images having noise
#works when characters are separated by some distance

import cv2
import numpy as np
from scipy import ndimage
import math
import scipy
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import numpy as np
from keras.preprocessing import image
import tensorflow as tf
from imutils.perspective import four_point_transform





def makesmall(gray_img):
	scale_percent = 50
	width = int(gray_img.shape[1] * scale_percent / 100)
	height = int(gray_img.shape[0] * scale_percent / 100)
	gray_img=cv2.resize(gray_img,(width,height))
	return gray_img


def preprocess(image):
	image=makesmall(image)
	thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
	opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
	# Find contours and remove small noise
	cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]
	for c in cnts:
		area = cv2.contourArea(c)
		if area < 50:
			cv2.drawContours(opening, [c], -1, 0, -1)

	return(opening)



def check(img):
	i=img.shape[0]
	f=0
	cnt=0
	for i in range(img.shape[0]//3,img.shape[0]):
		for j in range(img.shape[1]):
			if(img[i][j]==255):
				cnt+=1

	if(cnt>10):
		return(True)
	else:
		return(False)



def load_image(path):
    test_image = cv2.imread(path)
    gray = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray, (5,5), 0)
    return gray_img


def remove_noise_and_preprocess(img):
    gray_img = img.copy()
    gray_img=makesmall(gray_img)
    threshold=cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 20)
    return threshold


def word_segmentation(img):
    cpyimg=img.copy()
    for i in range(img.shape[0]):
        cnt=0
    for j in range(img.shape[1]):
        if(img[i][j]==255):
            cnt=cnt+1
        p=img.shape[1]
        percent=(100.0*cnt)/p
        if(percent>85):
            for j in range(img.shape[1]):
                img[i][j]=0
    kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=4)
    contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours.sort(reverse=True,key=cv2.contourArea)
    contour=contours[0]
    x,y,w,h = cv2.boundingRect(contour)
    cropped=cpyimg[y:y+h,x:x+w]
    return cropped



def getdist(x1,x2,y1,y2):
	return(math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)))



def houghtransform(img):
	newimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# Apply edge detection method on the image
	edges = cv2.Canny(img,50,150,apertureSize = 3)
	# This returns an array of r and theta values
	lines = cv2.HoughLines(edges,1,np.pi/180,img.shape[1]//10)
	# The below for loop runs till r and theta value
	px1=-1
	px2=-1
	py1=-1
	py2=-1
	if(lines is None):
		return px1,px2,py1,py2
	mxd=0
	for r,theta in lines[0]:
		
		# Stores the value of cos(theta) in a
		a = np.cos(theta)

		# Stores the value of sin(theta) in b
		b = np.sin(theta)
		
		# x0 stores the value rcos(theta)
		x0 = a*r
		
		# y0 stores the value rsin(theta)
		y0 = b*r
		
		# x1 stores the rounded off value of (rcos(theta)-1000sin(theta))
		x1 = int(x0 + 1000*(-b))
		
		# y1 stores the rounded off value of (rsin(theta)+1000cos(theta))
		y1 = int(y0 + 1000*(a))

		# x2 stores the rounded off value of (rcos(theta)+1000sin(theta))
		x2 = int(x0 - 1000*(-b))
		
		# y2 stores the rounded off value of (rsin(theta)-1000cos(theta))
		y2 = int(y0 - 1000*(a))
		curd=getdist(x1, x2, y1, y2)
		if curd>mxd:
			mxd=curd
			px1=x1
			px2=x2
			py1=y1
			py2=y2
	return (px1,px2,py1,py2)


def loadHINDImodel(path):
	model = tf.keras.models.load_model(path)
	return model




def predictchar(image,model):
	image=cv2.resize(image,(64,64))
	image=image*1/255.0
	image = np.expand_dims(image, axis=0)
	image = np.expand_dims(image, axis=3)
	lists = model.predict(image)[0]
	return np.argmax(lists)



def extractroi(img):
	cpyimg=img.copy()
	kernel = np.ones((2,2), np.uint8)
	img = cv2.dilate(img, kernel, iterations=2)
	contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	contours.sort(reverse=True,key=cv2.contourArea)
	contour=contours[0]
	x,y,w,h = cv2.boundingRect(contour)
	cropped=cpyimg[y:y+h,x:x+w]
	return cropped




