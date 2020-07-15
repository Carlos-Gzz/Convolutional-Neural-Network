# CNN
# Importing the libraries
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# Preprocessing the Training Set
train_datagen = ImageDataGenerator(
    rescale=1. / 255, # Feature scaling of the picture pixels.
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
training_set =  train_datagen.flow_from_directory(
    'C:\\Users\\...\\dataset\\training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary') # We select binary as we have a prediction between A or B, otherwise it would be categorical.

test_datagen = ImageDataGenerator(rescale=1./255)
test_set = test_datagen.flow_from_directory(
    'C:\\Users\\...\\dataset\\test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# Building the CNN
# Initialising the CNN
cnn = tf.keras.models.Sequential()

# Step 1 Convolution
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))

# Step 2 Pooling
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Adding a second convolutional layer
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

# Step 3 Flattening
cnn.add(tf.keras.layers.Flatten())

# Step 4 Full Connection
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))

# Step 5 Output Layer
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) # Sigmoid for binary classification, softmax for multilinear classification

# Training the CNN
# Compiling the CNN
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Training the CNN on the Training set and evaluating it on the Test set
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)

# Making a single prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:\\Users\\...\\dataset\\single_prediction\\cat_or_dog_1.jpg', target_size= (64, 64))
# This lines of code is to convert the image into an array
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
print(training_set.class_indices) # In order to know which one is A/B (in my case it was a dog/cat)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print(prediction)