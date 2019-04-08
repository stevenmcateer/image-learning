from keras.models import Sequential
from keras.layers import Dense, Activation
import tensorflow as tf
import numpy as np

# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
#
# Initialize weights randomly for each layer
# Experiment with ReLu Activation Units, as well as SeLu and Tanh
# Experiment with number of layers/num of neurons per layer
#
# Fill in Model Here
#
#
# Leave last layer the same
model.add(Dense(10, kernel_initializer='he_normal')) # last layer
model.add(Activation('softmax'))


# Compile Model
model.compile(optimizer='sgd',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train Model
history = model.fit(x_train, y_train,
                    validation_data = (x_val, y_val),
                    epochs=10,
                    batch_size=512)


# Report Results

print(history.history)
model.predict()
