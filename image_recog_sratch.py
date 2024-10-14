import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

dataset = pd.read_csv("insurance_data.csv")
x = dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values
#print(x.shape)
#print(y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset[['age', 'affordibility']], dataset.bought_insurance,test_size=0.2, random_state = 25)
#print(x_train)
x_train_scaled= x_train.copy()
x_train_scaled['age'] = x_train_scaled['age']/100
x_test_scaled= x_test.copy()
x_test_scaled['age'] = x_test_scaled['age']/100
print(x_train_scaled)
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(x_train_scaled, y_train, epochs=5000)
model.evaluate(x_test_scaled, y_test)


