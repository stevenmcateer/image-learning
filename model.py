from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np


def main():
    # Initialize weights randomly for each layer
    np.random.seed(0)

    # Load data
    images = preprocess_images(np.load("images.npy"))
    labels = preprocess_labels(np.load("labels.npy"))

    # Split data into each bin
    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    # Create the model
    model = build_model(training_images, training_labels, val_images, val_labels, 10)

    # Test the model
    test_model(model, testing_images, testing_labels)

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
    return keras.utils.to_categorical(labels, 10)

def split_data(images, labels):
    # Training set has 60%
    training_images = []
    training_labels = []
    training_length = 0.6*len(images)

    # Validation set has 15%
    val_images = []
    val_labels = []
    val_length = 0.15*len(images)

    # Test set has 25%
    testing_images = []
    testing_labels = []
    testing_length = 0.25*len(images)

    i = 0
    while i < len(images):
        random_num = np.random.randint(1,100)
        added = False
        while not added:
            if random_num <= 60:
                if len(training_images) != training_length:
                    added = True
                    training_images.append(images[i])
                    training_labels.append(labels[i])
                else:
                    random_num = np.random.randint(60,100)
            elif random_num <= 75:
                if len(val_images) != val_length:
                    added = True
                    val_images.append(images[i])
                    val_labels.append(labels[i])
                else:
                    random_num = np.random.randint(75,100)
            else:
                if len(testing_images) != testing_length:
                    added = True
                    testing_images.append(images[i])
                    testing_labels.append(labels[i])
                else:
                    random_num = np.random.randint(1,75)
        i+=1


    print("Length of training is", len(training_images), "should be", training_length)
    print("Length of validation is", len(val_images), "should be", val_length)
    print("Length of testing is", len(testing_images), "should be", testing_length)

    return (np.array(training_images), np.array(training_length)), \
           (np.array(val_images), np.array(val_length)), \
           (np.array(testing_images), np.array(testing_length))

def build_model(x_train, y_train, x_val, y_val, epochs):
    model = Sequential() # declare model
    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))

    # Experiment with ReLu Activation Units, as well as SeLu and Tanh
    # Experiment with number of layers/num of neurons per layer

    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # second layer
    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # third layer
    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # fourth layer
    #
    # Leave last layer the same
    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))

    # Compile Model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    y_train = keras.utils.to_categorical(y_train, 10)
    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        epochs=epochs,
                        batch_size=64)

    return model

def test_model(model, testing_images, testing_labels):
    return errors
#
#
# # Report Results
#
# print(history.history)
# model.predict()


if __name__ == '__main__':
    main()
