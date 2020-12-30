# DeepLearning-YogaPose-Classifier
OpenPose &amp; Deep Learning based model to classify Yoga Poses

1. LoadKaggleDataset.py
- To load dataset from Kaggle
- Containing 5 classes of yoga poses

2. PreProcessDataset.py
- To preprocess data for model training
- It extracts body keypoints from an image
- then, connects those to form body parts
- we obtain skeleton figures
- then, normalize and center the black skeleton on white canvas

3. ModelTraining.oy
- To train deep learning model for yoga pose classification of images
- Uses VGG16, with added custom layers
- Implements data augmentation

Refer to .pdf for visualizations
