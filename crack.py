# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 12:17:10 2022
project will be concrete crack image classification, 
you may apply transfer learning for this project.
@author: Farhan
credit: kong.kah.chun
"""

import numpy as np
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
import datetime
import os

#1. Load image files, then split into train-validation set
file_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\Dataset\Concrete_crack"
data_dir = pathlib.Path(file_path)

SEED = 12345
IMG_SIZE = (160,160)
BATCH_SIZE = 16

train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,
                                                            subset='training',seed=SEED,
                                                            shuffle=True,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,validation_split=0.3,
                                                            subset='validation',seed=SEED,
                                                            shuffle=True,
                                                            image_size=IMG_SIZE,
                                                            batch_size=BATCH_SIZE)
#%%
#2. Further split validation set,so that we obtain validation and test data
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches//5)
validation_dataset = val_dataset.skip(val_batches//5)

#Create prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE

train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#### Data preparation is completed ####

#%%
#3. Create model by applying transfer learning, we are using MobileNetV2 for this project
#3.1 Define a layer that preprocess inputs for the transfer learning model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#3.2 Create base model with MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False,
                                               wights='imagenet')

#3.3 Freeze the base model and view the model structure
base_model.trainable = False
base_model.summary()

#%%
#3.4 Create classification layers with global average pooling and dense layer
class_names = train_dataset.class_names
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
output_dense = tf.keras.layers.Dense(len(class_names), activation='softmax')

#3.5 Use functional API to build the entire model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x)
x = global_avg_pool(x)
outputs = output_dense(x)

model = tf.keras.Model(inputs, outputs)
#Print out the model structure
model.summary()

#%%
#3.6 Compile model
adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam, loss=loss, metrics=['accuracy'])

#%%
#4. Perform training
base_log_path = r"C:\Users\ACER\Desktop\SHRDC\Deep learning\TensorBoard\p3_log" 
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=2)

EPOCHS = 10
history = model.fit(train_dataset_pf,validation_data=validation_dataset_pf,
                   epochs=EPOCHS, callbacks=[tb_callback, es_callback])

#%%
#5. Evaluate with test dataset
test_loss, test_accuracy = model.evaluate(test_dataset_pf)

print('-------------------------Test Result------------------------')
print(f'Loss = {test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#6. Deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions, axis=1)

#%%
#7. Show some prediction results
plt.figure(figsize=(10,10))

for i in range(4):
    axs = plt.subplot(2,2,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f'Prediction: {current_prediction}, Actual: {current_label}')
    plt.axis('off')
    
save_path = r"C:\Users\ACER\Desktop\SHRDC\Git\AI-05 repo 3\Image"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')
plt.show()

