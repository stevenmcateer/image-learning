from model import split_data, preprocess_images, preprocess_labels
from sklearn import tree
import numpy as np



def main():
    # Load data
    images = preprocess_images(np.load("images.npy"))
    labels = preprocess_labels(np.load("labels.npy"))

    # Split data into each bin
    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    cl = tree.DecisionTreeClassifier()
    clf = cl.fit(training_images, training_labels)

    predicted = clf.predict(val_images)
    score = clf.score(val_images, val_labels)
    print(score)
    print(predicted)


if __name__ == "__main__":
    main()