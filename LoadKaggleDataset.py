'''
Python code to load Kaggle Dataset containing 5 Yoga Pose classes

Total training images-
	Downdog/  223 images
	Goddess/ 180 images
	Plank/ 266 images
	Tree/ 160 images
	Warrior/ 252 images

Total test images-
	Downdog/  97 images
	Goddess/ 80 images
	Plank/ 115 images
	Tree/ 69 images
	Warrior/ 109 images
'''

#I've used Google Colab for coding and Google Drive as workspace folder
from google.colab import drive
drive.mount('/content/gdrive')

import os
os.environ['KAGGLE_CONFIG_DIR'] = "/content/gdrive/My Drive/Colab Notebooks/Kaggle"

#changing the working directory
%cd /content/gdrive/My Drive/Colab Notebooks/Kaggle

#download dataset from Kaggle
!kaggle datasets download -d niharika41298/yoga-poses-dataset

#unzipping the zip files and deleting the zip files
!unzip \*.zip  && rm *.zip