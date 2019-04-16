Steven McAteer, William Schwartz

# Run

## to the CNN

```bash
$ python model.py
```

## to run the Decision Trees

```bash
$ python decision_tree.py
```

# Artificial Neural Network

## Description of Experiments

For the Artificial Neural Network, we experimented with many different parameters to achieve our
desired output. We experimented with the following values: number of dense layers, number of
neurons per layer, number of epochs, and batch size. See the Model & Training Project Description
section for more details.

## Model & Training Project Description

The first thing we experimented with was adding more layers. With each layer we added, we saw a
marginal improvement, but once we got to 6 total layers the performance started to decrease. We
decided that 5 layers seemed to be a good fit. The next alteration we made to the ANN was changing
the number of neurons per layer. At first we had 10 neurons for each, but we increased the number to 100.
This yielded better results, so we kept going and increased each layer to 500 neurons. The accuracy
at this level decreased, so we kept each level at 100 neurons, with the exception of the first layer
(28x28 neurons for each pixel) and the last layer (10 neurons). Next, we experimented with the number
of epochs per run; we found that 12 epochs seems to be the best. When increasing/decreasing the
number of epochs, performance suffered. Finally, we lowered the batch size significantly from 512 to
10. This increased performance significantly.


## Plot

![Accuracy/Validation Plot vs. Epochs](/acc_vs_val_plot.png)

## Visualization

The following 3 images were misclassified:

### Actual 1, Predicted 4
![Actual 1, Predicted 4](/actual_1_predicted_4.png)
### Actual 5, Predicted 3
![Actual 5, Predicted 3](/actual_5_predicted_3.png)
### Actual 8, Predicted 2
![Actual 8, Predicted 2](/actual_8_predicted_2.png)

## Model Performance & Confusion Matrix

The best performing model that we had for ANN has the following characteristics:

```
Number of layers: 5
Number of neurons per layer, respectively: (28*28), 100, 100, 100, 10
Epochs: 12
Batch Size: 10

```
### Confusion Matrix:
![Confusion Matrix](/ann_confusion_matrix.png)

```
Total Tests: 1625
Number of Accurate Labels: 1526
Accuracy: 0.939076923076923
```
![Precision/Recall](/precision_and_recall.png)

A copy of this trained model is saved as trained_model.h5


# Decision Trees

## Feature Extraction & Explanation

The features that we added to our decision tree algorithm are the following:

1. Average row: We averaged each row of the image
2. Average column: We averaged each column of the image
3. Average of the whole image: We averaged the entire image into one number
4. Compressed image: In this feature, we compressed the 28x28 image to a smaller,
14x14 image.

## Description of Experiments

For the decision trees, we altered a few parameter values in order to find the best results.
First of all, we specified the max depth to be 10. This increased performance because if the
max length is not set, it can result in over-fitting and become messy. Simplicity helped increase
accuracy in this scenario. The other parameter that we played with was the min_samples_leaf value.
This value is defined as the minimum number of samples needed to be considered a leaf node. We
tried limiting this to 10, but it decreased the performance. This means that by limiting the
growth of the tree, we could not yield better results.

Our features improved performance slightly, but we assume that all variations of the accuracy are
 due to random sampling.

## Model Performance and Confusion Matrix

### Baseline

Confusion Matrix:
![Confusion Matrix](/decision_tree_baseline_confusion_matrix.png)

```
Total Tests: 1625
Number of Accurate Labels: 1243
Accuracy: 0.7649230769230769
```
![Precision/Recall](/decision_tree_baseline_precision_and_recall.png)

The following 3 images were misclassified:

![Actual 1, Predicted 4](/decision_tree_baseline_actual_3_predicted_5.png)
![Actual 5, Predicted 3](/decision_tree_baseline_actual_5_predicted_2.png)
![Actual 8, Predicted 2](/decision_tree_baseline_actual_8_predicted_1.png)

### Restricted Tree Depth

Confusion Matrix:
![Confusion Matrix](/decision_tree_restricting_tree_depth_confusion_matrix.png)

```
Total Tests: 1625
Number of Accurate Labels: 1260
Accuracy: 0.7753846153846153
```
![Precision/Recall](/decision_tree_restricting_tree_depth_precision_and_recall.png)

The following 3 images were misclassified:

#### actual 1 predicted 3

![Actual 1, Predicted 4](/decision_tree_restricting_tree_depth_actual_1_predicted_3.png)

#### actual 3 predicted 9

![Actual 5, Predicted 3](/decision_tree_restricting_tree_depth_actual_3_predicted_9.png)

#### actual 5 predicted 9

![Actual 8, Predicted 2](/decision_tree_restricting_tree_depth_actual_5_predicted_9.png)

### Min Samples Leaf

Confusion Matrix:
![Confusion Matrix](/decision_tree_min_samples_leaf_confusion_matrix.png)

```
Total Tests: 1625
Number of Accurate Labels: 1167
Accuracy: 0.7181538461538461
```
![Precision/Recall](/decision_tree_min_samples_leaf_precision_and_recall.png)

The following 3 images were misclassified:

#### actual 1 predicted 8

![Actual 1, Predicted 4](/decision_tree_min_samples_leaf_actual_1_predicted_8.png)

#### actual 3 predicted 0

![Actual 5, Predicted 3](/decision_tree_min_samples_leaf_actual_3_predicted_0.png)

#### actual 4 predicted 0

![Actual 8, Predicted 2](/decision_tree_min_samples_leaf_actual_4_predicted_0.png)

### Custom Features

Confusion Matrix:
![Confusion Matrix](/decision_tree_features_confusion_matrix.png)

```
Total Tests: 1625
Number of Accurate Labels: 1273
Accuracy: 0.7833846153846153
```
![Precision/Recall](/decision_tree_features_precision_and_recall.png)

The following 3 images were misclassified:

#### actual 1 predicted 5

![Actual 1, Predicted 4](/decision_tree_features_actual_1_predicted_5.png)

#### actual 7 predicted 0

![Actual 5, Predicted 3](/decision_tree_features_actual_7_predicted_0.png)

#### actual 8 predicted 5

![Actual 8, Predicted 2](/decision_tree_features_actual_8_predicted_5.png)
