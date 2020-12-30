'''
Python code to train deep learning model to classify yoga poses
'''

import cv2
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

img_size = [200, 200]

#listing folers where train and test data is stored
train_data = 'train-processed'
valid_data = 'test-processed'

#fetching classes
num_classes = glob(train_data+'/*')
print(num_classes)
len(num_classes)

#data augmentation for increasing training dataset
train_gen = ImageDataGenerator(rescale=1./255,
                               shear_range = 0.2,
                               zoom_range = 0.2,
                               horizontal_flip = True)

#test data shouldn't be augmented
test_gen = ImageDataGenerator(rescale=1./255)

train_set = train_gen.flow_from_directory(train_data,
                                          target_size = (200,200),
                                          batch_size = 32,
                                          class_mode = 'categorical')

test_set = test_gen.flow_from_directory(valid_data,
                                          target_size = (200,200),
                                          batch_size = 32,
                                          class_mode = 'categorical')

#using VGG16 
vgg16 = VGG16(input_shape = img_size + [3], weights='imagenet', include_top=False)

#we don't want to retrain existing weights
for layer in vgg16.layers:
  layer.trainable = False

#add layer 
x = Flatten()(vgg16.output)
y = Dense(len(num_classes), activation='softmax')(x)
#creating model object
model = Model(inputs = vgg16.input, outputs = y)

#categorical cross entropy 
#because we are doing
#multiclass classification
model.compile(optimizer='adam', #gradient descent optimizer
              loss='categorical_crossentropy',
              metrics=['accuracy']) #to evaluate model
			  
#fitting model 
hist = model.fit(train_set, validation_data=test_set, epochs=5)

#it took 55 mins for fitting the model first time on 5 epochs, which achieved 90% accuracy

#plotting accuracy results
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#summarized history for loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#saving model to reuse
model.save("model.h5")

