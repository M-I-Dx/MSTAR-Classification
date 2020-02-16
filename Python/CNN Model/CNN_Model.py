# Deep Learning & ML libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Softmax, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
# Data analysis and handling libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
# Image preprocessing libraries
from PIL import Image
from imutils import paths
import cv2
from skimage.io import imread
# System & OS
import os
import sys
# Graph
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
# LIME
try:
    import lime
except:
    sys.path.append(os.path.join('..', '..')) # add the current directory
    import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
from lime.wrappers.scikit_image import SegmentationAlgorithm

def data_extraction(directory):
    images = []
    label = []
    num_images = 0
    for folder in os.listdir(directory):
        for filename in os.listdir("{}/{}".format(directory, folder)):
            img = cv2.imread(os.path.join(directory, folder, filename))
            print(num_images)
            if img is not None:
                images.append(img)
                label.append(folder)
                num_images += 1
    le = LabelEncoder()
    original_labels = label
    label = le.fit_transform(label)
    return images, label

directory_location = str(input("Enter directory location"))
data, labels = data_extraction(directory_location)
data = np.array(data)


# Save the images as np arrays, This helps in saving time during future use.
# np.save('data_numpy', data)
# np.save('label_numpy', labels)
# data = np.load('data_numpy.npy')
# labels = np.load('label_numpy.npy')

# This splits the data into training, validation and test set
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.20, stratify=labels)
(trainX, validateX, trainY, validateY) = train_test_split(trainX, trainY, test_size=0.20, stratify=trainY)

# This is our CNN model
model = Sequential()
model.add(Conv2D(8, (3, 3), input_shape=(200, 200, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(16, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))


model.add(Flatten())
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model settings
history = model.fit(trainX, trainY, epochs=10,shuffle=True,
                    validation_data=(validateX, validateY))

test_accuracy = model.evaluate(testX, testY, verbose=2)
pred_Y = model.predict_classes(testX)

# Plotting accuracy vs loss curve
plt.plot(history.history['acc'], label='acc')
plt.plot(history.history['val_acc'], label = 'val_acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.70, 1.01])
plt.legend(loc='lower right')
plt.savefig('Graph.png')


# Calculating various performance measures
labels_n ={'2S1': 0, 'BRDM-2': 1, 'BTR-60': 2, 'D7': 3, 'SLICY': 4, 'T62': 5, 'ZIL131': 6, 'ZSU-23/4': 7}
labels_nn = [0, 1, 2, 3, 4, 5, 6, 7]
num_examples = len(pred_Y)
C_values = np.zeros((2, 8, num_examples))
for i in range(8):
  C_values[0][:][i] = (pred_Y == labels_nn[i])
  C_values[1][:][i] = (testY == labels_nn[i])
preformance_measures = np.zeros((8, 6))
for i in range(8):
  TP = sum(((C_values[0][:][i] == 1) & (C_values[1][:][i] == 1)))
  FP = sum(((C_values[0][:][i] == 1) & (C_values[1][:][i] == 0)))
  TN = sum(((C_values[0][:][i] == 0) & (C_values[1][:][i] == 0)))
  FN = sum(((C_values[0][:][i] == 0) & (C_values[1][:][i] == 1)))
  precision = (TP)/(TP + FP)
  recall = (TP)/(TP + FN)
  f1_score = (2*precision*recall)/(recall + precision)
  specificity = (TN)/(TN + FP)
  ROC = ((recall**2 + specificity**2)**(0.5))/(2**(0.5))
  geometric_mean = (recall*specificity)**(0.5)
  preformance_measures[i][0] = precision
  preformance_measures[i][1] = recall
  preformance_measures[i][2] = f1_score
  preformance_measures[i][3] = specificity
  preformance_measures[i][4] = ROC
  preformance_measures[i][5] = geometric_mean


# Representing in Pandas DataFrame
preformance_measures = pd.DataFrame(preformance_measures, index=labels_n.keys(), columns=['Precision', 'Recall/Sensitivity', 'F1_Score', 'Specificity', 'ROC', 'Geometric Mean'] )
preformance_measures
preformance_measures.to_csv('CNN_Data.csv')

image_index = []
num_sample = 5
for i in range(num_sample):
    image_index.append(np.random.randint(len(testX) - 1))

# LIME explanation of the CNN model
explainer = lime_image.LimeImageExplainer()
segmenter = SegmentationAlgorithm('quickshift', kernel_size=1, max_dist=200, ratio=0.2)
explanations = []

x = 0
sample = []
for image_i in image_index:
    explanations.append(explainer.explain_instance(testX[image_i], model.predict, top_labels=2, hide_color=0, num_samples=10000, segmentation_fn=segmenter))
    sample.append(testX[image_i])
sample = np.array(sample)
model.predict(sample)

plt.figure(1, plt.figure(figsize=(100,100)))
index = 0
for explanation in explanations:
    temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=25, hide_rest=False)
    plt.subplot(20, 20, 1 + index)
    plt.imshow(mark_boundaries(temp, mask))
    index = index + 1
plt.show()
