from keras.models import Sequential
from keras.layers import Dense, Activation
import keras
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import matplotlib.colors as colors
from PIL import Image
from sklearn.model_selection import train_test_split
import random


def main():
    # Initialize weights randomly for each layer
    np.random.seed(0)

    # Number of runs
    epochs = 1

    # Load data
    images = preprocess_images(np.load("images.npy"))
    labels = preprocess_labels(np.load("labels.npy"))

    # Split data into each bin
    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    # Create the model
    model, history = build_model(training_images, training_labels, val_images, val_labels, epochs)

    # Save the model
    model.save("trained_model.h5")

    # Test the model
    errors = test_model(model, testing_images, testing_labels, history, epochs)

    # Save the misclassified images
    save_misclassified(errors)

def save_misclassified(errors):
    reshaped = []
    # Reshape the errors
    for e in errors[:4]:
        reshaped.append((np.uint8(e[0].reshape(28, 28)), e[1], e[2]))

    # Save each misclassified
    for r in reshaped:
        image = Image.fromarray(np.array(r[0]))
        image.save("actual_" + str(r[1]) + "_predicted_" + str(r[2]) + ".png")

def preprocess_images(images):
    # Flatten the matrices to 1x784 vectors
    flat_images = []
    for i in range(len(images)):
        flat_images.append(images[i].flatten())

    images = np.array(flat_images)
    images = images.astype('float32')
    images /= 255
    return images

def preprocess_labels(labels):
    return keras.utils.to_categorical(labels, 10)

def split_data(images, labels):
    stratified_seed = random.randint(0, 100)

    # make the STRATIFIED training set
    training_images, verify_images, training_labels, verify_labels = train_test_split(images,labels,stratify=labels,test_size=0.4, random_state=stratified_seed) # before model building

    # make the STRATIFIED testing and validation set
    val_images, testing_images, val_labels, testing_labels = train_test_split(verify_images,verify_labels,stratify=verify_labels,test_size=0.625, random_state=stratified_seed) # before model building


    return (np.array(training_images), np.array(training_labels)), \
           (np.array(val_images), np.array(val_labels)), \
           (np.array(testing_images), np.array(testing_labels))

def build_model(x_train, y_train, x_val, y_val, epochs):
    model = Sequential() # declare model
    model.add(Dense(28*28, input_shape=(28*28, ), kernel_initializer='he_normal')) # first layer
    model.add(Activation('relu'))

    # Experiment with ReLu Activation Units, as well as SeLu and Tanh
    # Experiment with number of layers/num of neurons per layer

    model.add(Dense(900, input_shape=(28*28, ), kernel_initializer='he_normal')) # second layer
    model.add(Dense(900, input_shape=(28*28, ), kernel_initializer='he_normal')) # third layer
    model.add(Dense(12, input_shape=(28*28, ), kernel_initializer='he_normal')) # fourth layer
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

    return model, history

def create_charts(history, epochs, confusion_matrix):
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    epoch_list = []
    for ep in range(epochs):
        epoch_list.append(ep)

    # Create Training/Val vs. epochs chart
    plot.plot(epoch_list, acc, 'go', epoch_list, val_acc, 'bo')
    plot.ylabel("Training Acc (Green), Validation Acc (Blue)")
    plot.xlabel("Epochs")
    plot.grid(True)
    plot.xticks(epoch_list)
    plot.savefig("acc_vs_val_plot.png")
    plot.close()


    # Create precision/recall chart
    precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    for num in range(len(confusion_matrix)):
        precision[num] = sum(row[num] for row in confusion_matrix)
        precision[num] = confusion_matrix[num][num] / precision[num]
        recall[num] = sum(confusion_matrix[num])
        recall[num] = confusion_matrix[num][num] / recall[num]

    print("Precision: ")
    precision_rounded = ["%.2f" % value for value in precision]
    print(precision_rounded)
    print("Recall: ")
    recall_rounded = ["%.2f" % value for value in recall]
    print(recall_rounded)

    f = open("ann_output.txt", "a")
    f.write("\nPrecision: \n" + str(precision_rounded))
    f.write("\nRecall: \n" + str(recall_rounded))
    f.close()

    rows = ["Precision", "Recall"]
    cols = [0, 1, 2 ,3 , 4, 5, 6, 7, 8, 9]
    fig, ax = plot.subplots()
    for item in [fig, ax]:
        item.patch.set_visible(False)
    vals = np.around([precision, recall], 2)
    normal = colors.Normalize(vals.min()-1, vals.max()+1)
    table = plot.table(cellText=[precision_rounded, recall_rounded],
        rowLabels=rows,
        colLabels=cols,
        loc='top',
        cellColours=plot.cm.Wistia(normal(vals)))
    table.scale(1, 5)
    plot.subplots_adjust(left=0.2, top=0.35)
    plot.axis('off')
    plot.savefig("precision_and_recall.png")
    plot.close()


def test_model(model, testing_images, testing_labels, history, epochs):
    errors = []

    # Predict and initialize matrix
    results = model.predict_on_batch(testing_images)
    confusion_matrix = [[0 for col in range(10)] for row in range(10)]

    i = 0
    num_correct = 0

    # Find how many labels are correct
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
            errors.append((testing_images[i]*255, a, r))

        i += 1

    # Create all charts and the confusion matrix visual
    create_confusion_matrix(confusion_matrix, results, num_correct)
    create_charts(history, epochs, confusion_matrix)


    return errors

def create_confusion_matrix(confusion_matrix, results, num_correct):
    # Print the confusion matrix
    print("Confusion Matrix:\n", np.array(confusion_matrix))
    print("Total Tests:", len(results),
		"\nNumber of Accurate Labels:", num_correct,
		"\nAccuracy:", num_correct/len(results))

    # Save all data to file
    f = open("ann_output.txt", "w+")
    f.write("Confusion Matrix:\n")
    f.write(str(np.array(confusion_matrix)))
    f.write("\nTotal Tests: " + str(len(results)))
    f.write("\nNumber of Accurate Labels: " + str(num_correct))
    f.write("\nAccuracy: " + str(num_correct/len(results)))


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



if __name__ == '__main__':
    main()
