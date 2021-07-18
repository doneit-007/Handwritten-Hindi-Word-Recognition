# Hnadwritten-Hindi-Word-Recognition
The aim of the project is to recognize single handwritten hindi word .

## RAW IMAGE
![image](https://user-images.githubusercontent.com/60650532/126056058-fddbca74-f114-493b-bb04-5f0d37ffd486.png)

## Task1 -Removal of Noises and Shadows from the raw image

We tried differnet thresholding techniques with appropirate parameter tuning  and came up doing bitwise and  of two images that we got  
1. After doing  Otsu Thresholding 
``` python
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
```
RESULT IMAGE<br/>
![image](https://user-images.githubusercontent.com/60650532/126056143-04d857be-980b-401d-92e4-3a0e8d71c784.png)



2. After doing adaptive thresholding
``` python
gray_img = img.copy()
    gray_img=makesmall(gray_img)
    threshold=cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 199, 20)
    return threshold
```
RESULT IMAGE<br/>

![image](https://user-images.githubusercontent.com/60650532/126056130-2199795d-60f7-4ca1-986f-a5ac987ed9e9.png)<br/><br/>
 FINAL IMAGE AFTER BITWISE AND <br/>
 ![image](https://user-images.githubusercontent.com/60650532/126056161-47df6cb0-4e41-4b62-b5f4-92439dd533b3.png)

It may look to you no much differnce between  adaptive ,otsu and final image here but there are many test images where it is giving very differnet results

## Task2 -To make the recognizer rotation invariant
It may happen that the given Hindi word is written in angled fashion like most of us use to do when we write on a board.
So to handle that we make use of _Houghtranform_ which is generally use to detect line .So we got all the line segment in image and out of that we find the paramters of the longest line segment.
``` python
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
```
Having the two pair of coordintes we can find the angle   of the longest line with horizontal to  rotate the image to get a horizontal word.
``` python
x1,x2,y1,y2=mosaic.houghtransform(new_img)
	angle=math.degrees(math.atan2(y2-y1, x2-x1))
	rot = ndimage.rotate(new_img, angle)
  ```
After Rotation image<br/>
![image](https://user-images.githubusercontent.com/60650532/126056551-0afcf12e-0a3a-4a3e-8c51-5ea9c0500f2b.png)<br/><br/>

## Task3 -Word Segmentation and Removing the header line
Now we got a clean and horizontal image so  we can extract the exact word region.This can be done simply by finding the rectangle of the maximum area that we got when we find contours in the image<br/>
Code looks like -:<br/>
``` python
def word_segmentation(img):
cpyimg=img.copy()
kernel = np.ones((5,5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=4)
    contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours.sort(reverse=True,key=cv2.contourArea)
    contour=contours[0]
    x,y,w,h = cv2.boundingRect(contour)
    cropped=cpyimg[y:y+h,x:x+w]
    return cropped
```
 As  we have a trained model which recognize only individual hindi letter thus we can't recognize whole Hindi word directly. So we have to extract characters out of it.
So to get seperate characters we  remove the header line with some image preocessing.<br/>
 So the approach is to find the row which is having the maximum number of white pixels and then convert it and all rows near to it into  black.<br/>
 Code looks like-:<br/>
 ``` python
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
```
After header line is removed -:<br/>
![image](https://user-images.githubusercontent.com/60650532/126057438-285e955d-9120-46fd-a468-133a732e76e7.png)
<br/>
## Task4 -Character Segmentation
Now let see how that removed header line image help us to get invidual character.<br/>
We now want separate region for each character so that we can have diiferent images for each character that can be loaded into the model for recognition of character.
The apporach we use to do so is -:
* Get a list of column numbers where each character is starting and lastly the end column of last character.
	* This can be done by finding the total number of white pixels in a whole column and if it greater than a thresh value<br/>
	then that column is part of a character.<br/>
Code for this -:
``` python
col_sum = np.zeros((dilated.shape[1]))
	col_sum = np.sum(dilated,axis=0)
	thresh=(0.08*dilated.shape[0])
	for i in range(1,dilated.shape[1]):
		if col_sum[i-1] <=thresh and col_sum[i] >thresh and col_sum[i+1] >thresh:
			start_char.append(i)
		elif col_sum[i-1]>thresh and col_sum[i] <=thresh and col_sum[i+1] <=thresh:
			end_char.append(i)

	start_char.append(end_char[-1])
```
* So our single character would lie in between two column number that are just neighbour to it.
* We get that image part for each character and after some more checking to verfify if is a complete character we append it to our final character list.:<br/>
Code for this -:
``` python
character = []
	for i in range(1,len(start_char)):
		roi = rot[:,start_char[i-1]:start_char[i]]
		h=roi.shape[1]
		w=roi.shape[0]
		roi=mosaic.extractroi(roi)
		roi=cv2.resize(roi,(180,180))
		if(mosaic.check(roi) and h>=30 and w>=30):
			character.append(roi)
```
<br/>
Appended charcaters for the above example-:<br/>
<p float="left">
  <img src="https://user-images.githubusercontent.com/60650532/126059049-e3258095-4dd2-46c6-8175-a798a3d6d682.png" width="100" />
  <img src="https://user-images.githubusercontent.com/60650532/126059066-54de159a-483e-403c-8426-d7254a6ba906.png" width="100" /> 
  <img src="https://user-images.githubusercontent.com/60650532/126059079-ace02d67-b448-4847-be25-af0b2d53f87c.png" width="100" />
  <img src="https://user-images.githubusercontent.com/60650532/126059090-837a3238-4919-495e-8b3f-baf40ebb3b1b.png" width="100" /> 
  <img src="https://user-images.githubusercontent.com/60650532/126059097-6667e38d-1449-44e9-89cb-b651c3b944d8.png" width="100" />
  <img src="https://user-images.githubusercontent.com/60650532/126059103-4feb72fd-8079-41c4-adc5-e75398208d24.png" width="100" /> 
 
</p>	

<br/>






