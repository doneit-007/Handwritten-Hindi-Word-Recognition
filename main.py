# Python program FOR Hindi CHARACTER RECOGNITION MOSAIC PS1
#TEAM ULTRONIX 
#TEAM LEADER-RISHAB ARYA

#our code is invariant to rotation,
#the word can be anywhere not necessary in middle,
#works on blurry images,
#works on images having noise
#works when characters are separated by some distance



mapping ={0:u'ठ',1:u'ड',2:u'त',3:u'थ',4:u'द',5:u'क',6:u'न',7:u'प',8:u'फ',9:u'म',10:u'य',11:u'र',12:u'व',13:u'स',14:u'क्ष',15:u'त्र',16:u'ज्ञ',17:u'घ',18:u'च',19:u'छ',20:u'ज'}




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
import all_functions_used as mosaic

modelpath="C:\\Users\\Dell\\Downloads\\FINALPS1\\model.h5"
model=mosaic.load_model(modelpath)






def predict(imagepath):
	img = mosaic.load_image(imagepath)
	print(img.shape)
	#this function is for handling shadows 
	org = mosaic.remove_noise_and_preprocess(img)
	#this function removes major noise in the background
	new_img=mosaic.preprocess(img)
	#finally taking bitwise and to get final image with shadows as well as noise removed
	for i in range(new_img.shape[0]):
		for j in range(new_img.shape[1]):
			if(org[i][j]==255 and new_img[i][j]==255):
				continue
			else:
				new_img[i][j]=0

	cv2.imshow("processed_image",new_img)
	cv2.waitKey(1000)

	#this code is for making the code rotation invariant
	#basically we find the maxlength line and take its angle to rotate image
	x1,x2,y1,y2=mosaic.houghtransform(new_img)
	angle=math.degrees(math.atan2(y2-y1, x2-x1))
	rot = ndimage.rotate(new_img, angle)
	rot = mosaic.word_segmentation(rot)
	cv2.imshow('Rotated_word_image',rot)
	cv2.waitKey(1000)
	dilated = rot.copy()
	start_char = []
	end_char = []
	#below code is for removing the header line and then segment each character
	row = np.zeros(dilated.shape[1])
	mxrow=0
	mxcnt=0
	kernel = np.ones((2,2), np.uint8)
	dilated = cv2.dilate(dilated, kernel, iterations=1)
	dilated = cv2.erode(dilated, kernel, iterations=1)
	for i in range(dilated.shape[0]):
		cnt=0
		for j in range(dilated.shape[1]):
			if(dilated[i][j]==255):
				cnt=cnt+1
		if(mxcnt<cnt):
			mxcnt=cnt
			mxrow=i
	print(dilated.shape[0])
	plus=dilated.shape[0]//10
	for i in range(0,mxrow+plus):
		dilated[i]=row
	cv2.imshow("HeaderLine Removed",dilated)
	cv2.waitKey(1000)
	col_sum = np.zeros((dilated.shape[1]))
	col_sum = np.sum(dilated,axis=0)
	thresh=(0.08*dilated.shape[0])
	for i in range(1,dilated.shape[1]):
		if col_sum[i-1] <=thresh and col_sum[i] >thresh and col_sum[i+1] >thresh:
			start_char.append(i)
		elif col_sum[i-1]>thresh and col_sum[i] <=thresh and col_sum[i+1] <=thresh:
			end_char.append(i)

	start_char.append(end_char[-1])
	character = []
	for i in range(1,len(start_char)):
		roi = rot[:,start_char[i-1]:start_char[i]]
		h=roi.shape[1]
		w=roi.shape[0]
		roi=mosaic.extractroi(roi)
		roi=cv2.resize(roi,(180,180))
		if(mosaic.check(roi) and h>=30 and w>=30):
			character.append(roi)
			cv2.imshow('CHARACTER_SEGMENTED', roi)
			cv2.waitKey(1000)

	ls=[]
	for char in character:
		pred=mosaic.predictchar(char, model)
		ls.append(mapping[pred])

	return ls







def test():
    image_paths = ['./test_images/s11.jpg']
    correct_answers = ['']
    score = 0
    multiplication_factor=2 
    for i,image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        answer = predict(image)
        print(''.join(answer))
        if correct_answers[i] == answer:
            score += len(answer)*multiplication_factor
    
    print('The final score of the participant is',score)





if __name__ == "__main__":
    test()
