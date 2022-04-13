# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:03:02 2022

@author: Farhan
credit: kong.kah.chun
"""

import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#1. Read CSV data
filepath = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\Dataset\Breast cancer\data.csv"
data_cancer = pd.read_csv(filepath)

#%%
#2. The column which is not useful as a feature will be remove
data_cancer = data_cancer.drop(['id','Unnamed: 32'], axis = 1)

#3. Split the data into features and label
cancer_features = data_cancer.copy()
cancer_label = cancer_features.pop('diagnosis')

#4. Check the split data
print("------------------Features--------------------")
print(cancer_features.head())
print("------------------Label-----------------------")
print(cancer_label.head())

#%%
#5. One hot encode label
#convert to number encoding
cancer_label_OH = pd.get_dummies(cancer_label)

#Check the one-hot label
print("---------------One-hot Label-----------------")
print(cancer_label_OH.head())

#6. Split the features and labels into train-validation-test sets (60:20:20 split)
SEED = 12345
x_train, x_iter, y_train, y_iter = train_test_split(cancer_features,cancer_label_OH,
                                                    test_size=0.4,random_state=SEED)
x_val, x_test, y_val, y_test = train_test_split(x_iter,y_iter,test_size=0.5,
                                                random_state=SEED)

#7. Normalize the features, fit with training data
scaler = StandardScaler()
scaler.fit(x_train)

x_train = scaler.transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

#### Data preparation is completed ####

#%%
#8. Create a feedforward neural network using TensorFlow Keras
number_input = x_train.shape[-1]
number_output = y_train.shape[-1]
model = tf.keras.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = number_input))
model.add(tf.keras.layers.Dense(64,activation='elu')) 
model.add(tf.keras.layers.Dense(32,activation='elu')) 
model.add(tf.keras.layers.Dropout(0.3)) 
model.add(tf.keras.layers.Dense(number_output,activation='softmax')) #output layer

#9. Compile the model
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#%%
#10. Train and evaluation of model
#Define callback functions: EarlyStopping and Tensorboard
base_log_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\TensorBoard\p1_log" 
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

EPOCHS = 100
BATCH_SIZE = 32
history = model.fit(x_train,y_train,validation_data=(x_val, y_val), batch_size=BATCH_SIZE,
                   epochs=EPOCHS, callbacks=[tb_callback, es_callback])

#%%
#Evaluate with test data for wild testing
test_result = model.evaluate(x_test,y_test,batch_size=BATCH_SIZE)
print(f"Test loss = {test_result[0]}")
print(f"Test accuracy = {test_result[1]}")

#%%
#Make prediction
predictions_softmax = model.predict(x_test)
predictions = np.argmax(predictions_softmax,axis=-1)
y_test_element, y_test_idx = np.where(np.array(y_test) == 1)
for prediction, label in zip(predictions,y_test_idx):
    print(f'Prediction: {prediction} Label: {label}, Difference: {prediction-label}')


