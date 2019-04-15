from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np
import matplotlib.pyplot as plot
import matplotlib.colors as colors


def main():
    # Initialize weights randomly for each layer
    np.random.seed(0)

    # Load data
    images = preprocess_images(np.load("images.npy"))
    labels = preprocess_labels(np.load("labels.npy"))

    # Split data into each bin
    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    # Create the model
    model = build_model(training_images, training_labels, val_images, val_labels, 12)

    # Test the model
    errors = test_model(model, testing_images, testing_labels)

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


    # print("Length of training is", len(training_images), "should be", training_length)
    # print("Length of validation is", len(val_images), "should be", val_length)
    # print("Length of testing is", len(testing_images), "should be", testing_length)


    return (np.array(training_images), np.array(training_labels)), \
           (np.array(val_images), np.array(val_labels)), \
           (np.array(testing_images), np.array(testing_labels))

def build_model(x_train, y_train, x_val, y_val, epochs):
    model = Sequential() # declare model
    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))

    # Experiment with ReLu Activation Units, as well as SeLu and Tanh
    # Experiment with number of layers/num of neurons per layer

    # model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # second layer
    # model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # third layer
    # model.add(Dense(28*12, input_shape=(28*28, ), kernel_initializer='he_normal')) # fourth layer
    #
    # Leave last layer the same
    model.add(Dense(10, kernel_initializer='he_normal')) # last layer
    model.add(Activation('softmax'))

    # Compile Model
    model.compile(optimizer='sgd',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # y_train = keras.utils.to_categorical(y_train, 3900)

    # Train Model
    history = model.fit(x_train, y_train,
                        validation_data = (x_val, y_val),
                        epochs=epochs,
                        batch_size=64)

    # Create the charts
    create_charts(history, epochs)

    return model

def create_charts(history, epochs):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']

    epoch_list = []
    for ep in range(epochs):
        epoch_list.append(ep)

    plot.plot(epoch_list, acc, 'ro', epoch_list, val_acc, 'bo')
    plot.ylabel("Training Acc, Validation Acc")
    plot.xlabel("Epochs")
    plot.grid(True)
    plot.xticks(epoch_list)
    plot.savefig("acc_vs_val_plot.png")
    plot.close()


def test_model(model, testing_images, testing_labels):
    errors = []

    # Predict and intialize matrix
    results = model.predict_on_batch(testing_images)
    confusion_matrix = [[0 for col in range(10)] for row in range(10)]

    i = 0
    num_correct = 0

    # Find how many label are correct
    testing_labels = testing_labels.tolist()
    for result in results:
        actual = testing_labels[i]
        result = result.tolist()
        r = result.index(max(result))
        a = actual.index(max(actual))

        confusion_matrix[a][r] +=1

        if a == r:
            num_correct += 1
        else:
            errors.append(testing_images[i])

        i += 1


    # Print the confusion matrix
    print("Confusion Matrix:\n", np.array(confusion_matrix))
    print("Total Tests:", len(results),
		"\nNumber of Accurate Labels:", num_correct,
		"\nAccuracy:", num_correct/len(results))

    # Save the matrix
    rows_and_cols = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fig, ax = plot.subplots()
    for item in [fig, ax]:
        item.patch.set_visible(False)

    vals = np.around(confusion_matrix, 2)
    normal = colors.Normalize(vals.min()-1, vals.max()+1)

    table = plot.table(cellText=confusion_matrix,
        rowLabels=rows_and_cols,
        colLabels=rows_and_cols,
        loc='top',
        cellColours=plot.cm.Wistia(normal(vals)))
    table.scale(1, 5)
    plot.subplots_adjust(left=0.2, top=0.35)
    plot.axis('off')
    plot.savefig("ann_confusion_matrix.png")
    plot.close()


    return errors



if __name__ == '__main__':
    main()
