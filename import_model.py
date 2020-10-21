import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow import keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense

import os.path

# Directory of model
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
my_model = os.path.join(THIS_FOLDER, "saved_models", 'medical_trial_model.h5')

new_model = load_model(my_model)

new_model.summary()
print(new_model.get_weights())
print(new_model.optimizer)

# Load json model 
#model_architecture = model_from_json(json_string)
#model_architecture.model_architecture()

# Import model weights 
model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

my_weights = os.path.join(THIS_FOLDER, "saved_models", 'my_model_weights.h5')
model2.load_weights(my_weights)

print(model2.get_weights())