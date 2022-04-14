# Breast Cancer Winconsin - Classification and Prediction

## 1. Summary of the project
This is Data set to classify the Benign and Malignant cells in the given dataset using the description about the cells in the form of columnar attributes. The aim of this project is to create a highly accurate deep learning model to predict breast cancer (whether the tumour is malignant or benign). The model is trained with [Winsonsin Breast Cancer Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

The breast cancer data includes 569 examples of cancer bipsies, each with 32 features. One feature is an identification number, another is the cancer diagnosis and 30 are numerical values laboratory measurements. The diagnosis is coded as "M" to indicate malignant or "B" to indicate benign

## 2. IDE and Framework
This project is created using Spyder as the main IDE. The main frameworks used in this project are:
- Pandas
- Scikit-learn
- TensorFlow Keras

## 3. Methodology

### _3.1 Data Pipeline_
The data is first loaded and preprocessed, such that unwanted features are removed, and label is encoded in one-hot format. Then the data is split into train-validation-test sets, with a ratio of 60:20:20.

### _3.2 Model Pipeline_
A feedforward neural network is constructed that is catered for classification problem. The structure of the mofel is fairy simple. Figure below shows the structure of the model.

![alt text](https://github.com/paan234/ai05-test-repo/blob/master/Image/model.png)

The model is trained with a batch size of 32 and for 100 epochs. Early stopping is applied in this training. The training stops at epoch 23, with a training accuracy of 99% and validation accuracy of 95%. The two figures below show the graph of the training process.

![alt text](https://github.com/paan234/ai05-test-repo/blob/master/Image/Loss_graph.png)

![alt text](https://github.com/paan234/ai05-test-repo/blob/master/Image/Accuracy_graph.png)

## 4. Results
Upon evaluating the model with test data, the model obtain the following test results, as shown in figure below.

![alt text](https://github.com/paan234/ai05-test-repo/blob/master/Image/Test_result.jpg)
