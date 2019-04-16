from model import split_data, preprocess_images, preprocess_labels
from sklearn import tree
import scipy.misc
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plot
import matplotlib.colors as colors
from PIL import Image
import statistics



def main():

    original_images = np.load("images.npy")
    original_labels = np.load("labels.npy")

    baseline_dt(original_images, original_labels)
    restricting_tree_depth(original_images, original_labels)
    min_samples_leaf(original_images, original_labels)

    features_dt(original_images, original_labels)


def baseline_dt(original_images, original_labels):
    # Split data into each bin

    images = preprocess_images(original_images)
    labels = preprocess_labels(original_labels)

    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    cl = tree.DecisionTreeClassifier()
    clf = cl.fit(training_images, training_labels)

    clf.predict(val_images)
    clf.score(val_images, val_labels)
    errors = test_model(cl, testing_images, testing_labels, "decision_tree_baseline" )
    save_misclassified(errors, "decision_tree_baseline")

def  restricting_tree_depth(original_images, original_labels):

    images = preprocess_images(original_images)
    labels = preprocess_labels(original_labels)

    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    cl = tree.DecisionTreeClassifier(max_depth=10)
    clf = cl.fit(training_images, training_labels)

    clf.predict(val_images)
    clf.score(val_images, val_labels)
    errors = test_model(cl, testing_images, testing_labels, "decision_tree_restricting_tree_depth" )
    save_misclassified(errors, "decision_tree_restricting_tree_depth")


def min_samples_leaf(original_images, original_labels):

    images = preprocess_images(original_images)
    labels = preprocess_labels(original_labels)

    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(images, labels)

    cl = tree.DecisionTreeClassifier(min_samples_leaf=10)
    clf = cl.fit(training_images, training_labels)

    clf.predict(val_images)
    clf.score(val_images, val_labels)
    errors = test_model(cl, testing_images, testing_labels, "decision_tree_min_samples_leaf" )
    save_misclassified(errors, "decision_tree_min_samples_leaf")
    

def features_dt(original_images, original_labels):
    compress = preprocess_images(average_chunk(original_images, 14, 14))
    all_avg = average_all(original_images)
    avg_row = average_row(original_images)
    avg_col = average_column(original_images)

    images = preprocess_images(original_images)
    labels = preprocess_labels(original_labels)
    features = np.concatenate((images,compress, all_avg.T, avg_row, avg_col), axis=1)

    (training_images, training_labels), (val_images, val_labels), (testing_images, testing_labels) = split_data(features, labels)

    cl = tree.DecisionTreeClassifier(min_samples_leaf=10)
    clf = cl.fit(training_images, training_labels)

    clf.predict(val_images)
    clf.score(val_images, val_labels)
    errors = test_model(cl, testing_images, testing_labels, "decision_tree_features" )
    save_misclassified(errors, "decision_tree_features")


     

def average_all(original_images):
    the_avg = []
    for img in average_row(original_images):
        the_avg.append(statistics.mean(img))
    return np.array([the_avg])


def average_row(original_images):
    images = []
    for image in original_images:
        row_avg = []
        for row in image:
            row_avg.append(statistics.mean(row))
        images.append(row_avg)
    return np.array(images)
    

def average_column(original_images):
    images = []
    for image in original_images:
        image = np.rot90(image)
        row_avg = []
        for row in image:
            row_avg.append(statistics.mean(row))
        images.append(row_avg)
    return np.array(images)

def cell_averaging(original_images):
    images = []
    for image in original_images:
        image = np.rot90(image)
        row_avg = []
        for row in image:
            row_avg.append(statistics.mean(row))
        images.append(row_avg)
    return np.array(images)

def average_chunk(original_images, i_height, i_width):
    images = []
    for image in original_images:
        images.append(np.array(Image.fromarray(image).resize((i_height, i_width))))

    return images




def test_model(model, testing_images, testing_labels, dt_name):
    errors = []

    # Predict and initialize matrix
    results = model.predict(testing_images)
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
            errors.append((testing_images[i][0:784]*255, a, r))

        i += 1

    # Create all charts and the confusion matrix visual
    create_charts(confusion_matrix, dt_name)
    create_confusion_matrix(confusion_matrix, results, num_correct, dt_name)

    return errors


def create_charts(confusion_matrix, dt_name):

    # Create precision/recall chart
    precision = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    recall = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


    f = open("{}.txt".format(dt_name) ,"w")

    for num in range(len(confusion_matrix)):
        precision[num] = sum(row[num] for row in confusion_matrix)
        precision[num] = confusion_matrix[num][num] / precision[num]
        recall[num] = sum(confusion_matrix[num])
        recall[num] = confusion_matrix[num][num] / recall[num]


    print("Precision: ")
    f.write("Precision: \n")
    precision_rounded = ["%.2f" % value for value in precision]
    print(precision_rounded)
    f.write(str(precision_rounded)+"\n")
    print("Recall: ")
    f.write("Recall: \n")
    recall_rounded = ["%.2f" % value for value in recall]
    print(recall_rounded)
    f.write(str(recall_rounded)+"\n")

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
    plot.savefig("{}_precision_and_recall.png".format(dt_name))
    plot.close()


def create_confusion_matrix(confusion_matrix, results, num_correct, dt_name):
    # Print the confusion matrix

    f = open("{}.txt".format(dt_name) ,"a")
    f.write("Confusion Matrix:\n")
    f.write(str(np.array(confusion_matrix)))
    f.write("\nTotal Tests: " + str(len(results)))
    f.write("\nNumber of Accurate Labels: " + str(num_correct))
    f.write("\nAccuracy: " + str(num_correct/len(results)))


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
    plot.savefig("{}_confusion_matrix.png".format(dt_name))
    plot.close()

def save_misclassified(errors, dt_name):
    reshaped = []
    # Reshape the errors
    for e in errors[:4]:
        reshaped.append((np.uint8(e[0].reshape(28, 28)), e[1], e[2]))

    # Save each misclassified
    for r in reshaped:
        image = Image.fromarray(np.array(r[0]))
        image.save("{}_actual_".format(dt_name) + str(r[1]) + "_predicted_" + str(r[2]) + ".png")


if __name__ == "__main__":
    main()
