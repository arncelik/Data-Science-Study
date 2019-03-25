#!/usr/bin/env python
# coding: utf-8

# # Implementing LeNet with Python and Keras

# In[ ]:


#import the necessary packages
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K


# In[15]:


class LeNet:
    @staticmethod
    def build(numChannels, imgRows, imgCols, numClasses,
             activation='relu', weightsPath=None):
        #initialize the model
        model = Sequential()
        inputShape = (ingRows,imgCols, numChannels)
        
        #if we are using "channles first", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)
        #define the first set of CONV => Activation -> Pool layers
        model.add(Conv2D(20,5,padding='same',input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2, 2)))
        #define the first FC => Activation layers
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))
    
        #defines the second FC layer
        model.add(Dense(num(Classes)))
    
        #lastly, define the soft-max classifier
        model.add(Activation("softmax"))
        
        if weightsPath is not None:
            model.load_weights(weightsPath)
            
        #return the constructed network architecture
        return model


# In[2]:


#import the necessary packages
from pyimagesearch.cnn.networks.lenet import LeNet
from sklearn.mmodel_selection import train_test_split
from leras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backedn as K
import numpy as np
import argparse
import cv2 as cv


# In[1]:


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1,
    help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
    help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
    help="(optional) path to weights file")
args = vars(ap.parse_args())


# In[ ]:


print("[INFO] downloading MNIST...")
((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0


# In[2]:


trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)
 
print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1, imgRows=28, imgCols=28,
    numClasses=10,
    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt,
    metrics=["accuracy"])


# In[3]:


if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=128, epochs=20,
        verbose=1)
 
    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
        batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))


# In[ ]:


if args["save_model"] > 0:
    print("[INFO] dumping weights to file...")
    model.save_weights(args["weights"], overwrite=True)


# In[ ]:


for i in np.random.choice(np.arange(0, len(testLabels)), size=(10,)):
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis=1)
 
   
    if K.image_data_format() == "channels_first":
        image = (testData[i][0] * 255).astype("uint8")
 
    else:
        image = (testData[i] * 255).astype("uint8")
 

    image = cv2.merge([image] * 3)

    image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
 
    # shows the image and prediction
    cv2.putText(image, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}, Actual: {}".format(prediction[0],
        np.argmax(testLabels[i])))
    cv2.imshow("Digit", image)
    cv2.waitKey(0)

