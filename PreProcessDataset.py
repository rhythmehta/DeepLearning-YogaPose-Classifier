'''
Python Code to preprocess data for model trainng
'''

#Using Google Colab for coding and Google Drive for workspace folder
from google.colab import drive 
drive.mount('/content/gdrive')

#changing working directory for TRAIN set
%cd /content/gdrive/My Drive/Colab Notebooks/Kaggle/DATASET/TRAIN

#to list folders in working directory
!ls

#loading folders containing training data images
#len(class_name) can be used after importing to get the count
#of images available per class
from glob import glob

#loading class-wise data
downdog = glob("downdog/*")
goddess = glob("goddess/*")
plank = glob("plank/*")
tree = glob("tree/*")
warrior2 = glob("warrior2/*")
classes = [downdog, goddess, plank, tree, warrior2]

#importing OpenPose MPII model trained on deep neural networks
#the 2 files below can be obtained on Github@CMU-Perceptual-Computing-Lab
#the following model can help us extract human body key points
#refer to pdf for corresponding body part to each key point
protoFile = "pose_deploy_linevec_faster_4_stages.prototxt.txt"
weightsFile = "pose_iter_160000.caffemodel"

#total keypoints
nPoints = 15

#sensible connection of keypoints that form a body part
#Eg., Right Elbow – 3, Right Wrist – 4; pairing them gives us right forearm
POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

#keypoint classification probability threshold, for us to use a keypoint
threshold = 0.1

#loading deep neural network OpenPose MPII
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

#image dimensions for the network
inWidth = 368
inHeight = 368

#importing necessary libraries we're gonna need forward
from google.colab.patches import cv2_imshow
from datetime import datetime 
import numpy as np
import cv2

startTime = datetime.now()
n = 0

for pose_class in classes:
	for pic in pose_class: #folder containing all the images
	  
	  frame = cv2.imread(pic) #loading the pic
	  frameWidth, frameHeight = frame.shape[1], frame.shape[0]
	  
	  #creating a blank image of same dimension
	  img = np.zeros([frameHeight, frameWidth, 3], dtype=np.uint8) 
	  img.fill(255) #filling with white color
	  
	  #preparing pic for input to MPII model
	  inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), 
									  (0, 0, 0), swapRB=False, crop=False)
	  
	  net.setInput(inpBlob) #giving the image as an input to the network
	  
	  output = net.forward() #receiving output
	  
	  #height, width of the output
	  H, W = output.shape[2], output.shape[3] 

	  #empty list to store the detected keypoints
	  points = []

	  for i in range(nPoints):
		  #confidence map of corresponding body's part.
		  probMap = output[0, i, :, :]

		  #find global maxima of the probMap.
		  minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

		  #scaling the point to fit on the original image
		  x = (frameWidth * point[0]) / W
		  y = (frameHeight * point[1]) / H

		  if prob > threshold: 
			  #append point to the list if the probability > threshold
			  #append point to the list if the probability > threshold
			  points.append((int(x), int(y)))
		  else :
			  points.append(None)
	  
	  if points[14]: #to check if point 14 (chest) is detected/estimated
		
		#getting the image center point
		center = (frame.shape[1]//2, frame.shape[0]//2) 
		
		#finding how much we need to shift points to align
		#body center with image center 
		shift = np.subtract(points[14], center) 
		
		newPoints = [(0,0) for _ in range(len(points))]
		for i in range(len(points)):
		  if points[i]:
			newPoints[i] = tuple(np.subtract(points[i], shift))
		  else:
			newPoints[i] = None
		
		for pair in POSE_PAIRS: #plotting skeleton on the blank white image 
		  partA = pair[0]
		  partB = pair[1]

		  if newPoints[partA] and newPoints[partB]:
			  #joining them with a black line
			  cv2.line(img, newPoints[partA], newPoints[partB], (0, 0, 0), 2)
		n+=1
		#saving processed image
		cv2.imwrite('str(pose_class)'+'_processed/'+'str(pose_class)'+str(n)+'.jpg', img)
		#even though I am saving them in a processed folder above
		#later I actually moved them in my drive hence 
		#the folder url look different in modeltraining.py

#it took me ~25 mins per train class & ~10 mins per test class for preprocessing
#similarly, preprocess test set for the sake of model metrics
#in application, each image will be required to first preprocess then 
#class prediction via the our trained model
