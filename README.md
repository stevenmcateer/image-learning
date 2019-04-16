Steven McAteer, William Schwartz

# Artificial Neural Network

## Description of Experiments


## Model & Training Project Description

For this algorithm, we experimented with many different parameters to achieve our desired output.
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

![Actual 1, Predicted 4](/actual_1_predicted_4.png)
![Actual 5, Predicted 3](/actual_5_predicted_3.png)
![Actual 8, Predicted 2](/actual_8_predicted_2.png)

## Model Performance & Confusion Matrix

The best performing model that we had for ANN has the following characteristics:

Number of layers: 5
Number of neurons per layer, respectively: (28*28), 100, 100, 100, 10
Epochs: 12
Batch Size: 10

Confusion Matrix:
![Confusion Matrix](/ann_confusion_matrix.png)

Total Tests: 1625
Number of Accurate Labels: 1526
Accuracy: 0.939076923076923
Precision:
['0.96', '0.96', '0.94', '0.96', '0.90', '0.94', '0.95', '0.93', '0.93', '0.93']
Recall:
['0.98', '0.97', '0.92', '0.89', '0.95', '0.91', '0.98', '0.95', '0.94', '0.89']

A copy of this trained model is saved as trained_model.h5



# Decision Trees

## Feature Extraction & Explanation

The features that we added to our decision tree algorithm are the following:

1. Average row totals:
2. Average column totals:
3. Average of the whole image:
4. Compressed image (14x14):

## Description of Experiments

For the decision trees, we altered a few parameter values in order to find the best results.
First of all, we specified the max depth to be 10. This increased performance because if the
max length is not set, it can result in over-fitting and become messy. Simplicity helped increase
accuracy in this scenario. The other parameter that we played with was the min_samples_leaf value.
This value is defined as the minimum number of samples needed to be considered a leaf node. We
tried limiting this to 10, but it decreased the performance. This means that by limiting the
growth of the tree, we could not yield better results.

## Model Performance and Confusion Matrix

The best performing model that we had for decision trees has the following characteristics:

Confusion Matrix:
![Confusion Matrix](/dt_confusion_matrix.png)







