import tensorflow as tf
from tensorflow import keras
###Use the "ImageDataGenerator()" class from keras.processing.image to build out an instance called "train_datagen" with the following parameters:

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip=True)

### Then build your training set by using the method ".flow_from_directory()"
train = train_datagen.flow_from_directory(
    directory = 'file path',
    target_size = (64, 64),
    batch_size = 32,
    class_mode = 'categorical')

### Take a look at your training set:

X_train,y_train = train.next() 
for i in range(0,X_train.shape[0]):
    image = X_train[i]
    print(" Image shape of " + str(i) + "th Observation: ",image.shape)
input_shape = X_train[0].shape

NumberofClasses = len(y_train[0])
print("Number of classes to predict on: " , NumberofClasses)

## Initial Classifier Build:
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

# Create an instance of Sequential called "classifier"
classifier_1 = Sequential() 

# Add a Conv2D layer
classifier_1.add(Conv2D(32, (2, 2), input_shape=input_shape)) 
classifier_1.add(Activation('relu')) 

# Add a MaxPooling2D layer
classifier_1.add(MaxPooling2D(pool_size=(2, 2))) 

# Add another Conv2D layer
classifier_1.add(Conv2D(64, (3, 3), input_shape=input_shape)) 
classifier_1.add(Activation('relu')) 

# Add a MaxPooling2D layer
classifier_1.add(MaxPooling2D(pool_size=(2, 2))) 

# Add a Flatten layer
classifier_1.add(Flatten()) 

# Add a Dense layer
classifier_1.add(Dense(128)) 
classifier_1.add(Activation('relu')) 

# Add a final Dense layer (this will output our probabilities)
classifier_1.add(Dense(NumberofClasses)) 
classifier_1.add(Activation('softmax')) 

# Compiling
classifier_1.compile(loss='categorical_crossentropy', 
                   optimizer='adam', 
                   metrics=['accuracy']) 


### Use .fit() with the training set. For the first run, use the following parameters
classifier_1.fit(
    train,
    epochs=3,
    steps_per_epoch=3
)



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model

### save model to a file

classifier_1.save('model_1.h5')

### Predict using the model built

import os, glob
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# returns a compiled model
# identical to the previous one
model = load_model('model_1.h5')
print("Loaded model from disk")

# test data path
img_dir = "/Users/kazotogbah/Desktop/MachineLearningPredictiveAnalytics/ML_session_7/dataset_test" # Enter Directory of test set

# iterate over each test image
data_path = os.path.join(img_dir, '*g')
files = glob.glob(data_path)

# print the files in the dataset_test folder 
for f in files:
    print(f)
    
# make a prediction and add to results 
data = []
results = []
for f1 in files:
    img = image.load_img(f1, target_size = (64, 64))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    data.append(img)
    result = model.predict(img)
    r = np.argmax(result, axis=1)
    results.append(r)

results


## determine accuracy

# Check category labels in training_set
train.class_indices

# Create a list to store the category/labels for the test data as the actual values. 
actual_test = [3, 0, 2, 0, 1, 2, 3, 1]

# Compare the predicted values to the actual values for the test set and calculate accuracy score
from sklearn.metrics import accuracy_score
ped_test = np.array(results).flatten().tolist()
accuracy_score(pred_test, actual_test)

