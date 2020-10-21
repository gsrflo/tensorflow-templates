import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler 
import itertools
import os.path

## for tensorflow sequential model
# for model building
import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
# for model training
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# for confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

## if GPU available
#physical_devices =  tf.config.experimental.list_physical_devices('GPU')
#print("Num GPUs availagle: ", len(physical_devices))
#tf.config.experimental.set_memory_growth(physical_devices[0], True)


# create empty list
train_labels = []
train_samples = []

## create sample data
for i in range(50):
    # the ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # the ~5% of older individuals who did not experience side effects 
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0)

for i in range(1000):
    # the ~95% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # the ~95% of older individuals who did not experience side effects 
    random_older = randint(65,100)
    train_samples.append(random_older)
    train_labels.append(1)


# create numpy arrays an shuffle data, because fit function expects np array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# normalize data: training becomes quicker and more efficient
scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1))    # fit function does not accept 1D data

# build sequential model: dense layer = fully connected layer
model = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),   #1st hidden layer
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')                    #output layer, 2 possible output classes, softmax gives output probablilty
])

# print model summary
model.summary()

# for compiling model and making it ready for training
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# train model
model.fit(x=scaled_train_samples, y=train_labels, validation_split=0.1, batch_size=10, epochs=30, shuffle=True, verbose=2)
# batch: how many samples are included to be passed and processed by the network in one time
# epoch: model is going to train all the data 30 times
# shuffle: by default true, so that any order inside the data gets erased (happens after validation_split)
# verbose: option to see output {1,2}
# validation_split: percantage of data which is going to be used for validation 
# validation_data: handover list with data for validation


## Creating a test set
test_labels = []
test_samples = []

# create sample data
for i in range(10):
    # the ~5% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # the ~5% of older individuals who did not experience side effects 
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(0)

for i in range(200):
    # the ~95% of younger individuals who did experience side effects
    random_younger = randint(13,64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # the ~95% of older individuals who did not experience side effects 
    random_older = randint(65,100)
    test_samples.append(random_older)
    test_labels.append(1)


# create numpy arrays an shuffle data, because fit function expects np array
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples)

# normalize data: training becomes quicker and more efficient
scaler = MinMaxScaler(feature_range=(0,1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))    # fit function does not accept 1D data


## Predict on test data
predictions = model.predict(x=scaled_test_samples, batch_size=10, verbose=0)

#output predictions
for i in predictions:
    print(i)

# binary output
rounded_predictions = np.argmax(predictions, axis=-1)
for i in rounded_predictions:
    print(i)


## Create confusion matrix 
cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

cm_plot_labels = ['no_side_effects', 'had_side_effects']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cm_plot_labels)
disp = disp.plot()
plt.show()

# Save model architecture, weights, training configuration, state of the optimizer
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_model = os.path.join(THIS_FOLDER, "saved_models",'medical_trial_model.h5')

if os.path.isfile(my_model) is False:
     model.save(my_model)

# Save model as json: saves just architecture of model without weights, etc.
json_string = model.to_json()
print(json_string)

# Save model weights 
my_model_weights = os.path.join(THIS_FOLDER, "saved_models", 'my_model_weights.h5')

if os.path.isfile(my_model_weights) is False:
     model.save_weights(my_model_weights)      