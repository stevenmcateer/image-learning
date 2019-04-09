from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


def main():
    # Initialize weights randomly for each layer
    np.random.seed(0)
    images = preprocess_images(np.load("images.npy"))
    labels = preprocess_labels(np.load("labels.npy"))


def preprocess_images(images):
    # Flatten the matrices to 1x784 vectors
    flat_images = []
    for i in range(len(images)):
        flat_images.append(images[i].flatten())

    images = np.array(flat_images)
    # images = images.astype('float32')
    # images /= 255
    return images

def preprocess_labels(labels):
    return np_utils.to_categorical(labels, 10)

# def split_data():
#     # Training set has 60%
#
#
#     # Validation set has 15%
#
#     # Test set has 25%


# Model Template

model = Sequential() # declare model
model.add(Dense(10, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
model.add(Activation('relu'))
#
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


if __name__ == '__main__':
    main()
