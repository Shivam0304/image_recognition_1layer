import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

dataset = pd.read_csv("insurance_data.csv")
x = dataset.iloc[:,:-1].values
y= dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(dataset[['age', 'affordibility']], dataset.bought_insurance,test_size=0.2, random_state = 25)

x_train_scaled= x_train.copy()
x_train_scaled['age'] = x_train_scaled['age']/100
x_test_scaled= x_test.copy()
x_test_scaled['age'] = x_test_scaled['age']/100

'''print(x_train_scaled)
model = keras.Sequential([
    keras.layers.Dense(1, input_shape=(2,), activation='sigmoid', kernel_initializer='ones', bias_initializer='zeros')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train_scaled, y_train, epochs=5000)
model.evaluate(x_test_scaled, y_test)

'''
def sigmoid_numpy(X):
   return 1/(1+np.exp(-X))



def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_new = [max(i,epsilon) for i in y_predicted]
    y_predicted_new = [min(i,1-epsilon) for i in y_predicted_new]
    y_predicted_new = np.array(y_predicted_new)
    return -np.mean(y_true*np.log(y_predicted_new)+(1-y_true)*np.log(1-y_predicted_new))

class insur_pred:
    def _init_(self):
        self.w1=1
        self.w2=1
        self.bias=0
    
    def predict(self,x_test):
        y_predicted=sigmoid_numpy(self.w1*x_test['age'] + self.w2*x_test['affordibility'] + self.bias)
        return y_predicted
    
    def fit(self, x_test, y_true, epochs, loss_thresold):
        self.w1, self.w2, self.bias = self.gradient_descent(x_test['age'], x_test['affordibility'],y_true, epochs, loss_thresold)
        print(f'final-->w1:{self.w1}, final-->w2:{self.w2}, final-->bias:{self.bias}')

    def gradient_descent(self, age,affordibility, y_true, epochs, loss_thresold):
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)
        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordibility + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)
            
            w1d = (1/n)*np.dot(np.transpose(age),(y_predicted-y_true)) 
            w2d = (1/n)*np.dot(np.transpose(affordibility),(y_predicted-y_true)) 

            bias_d = np.mean(y_predicted-y_true)
            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d
            
            if i%50==0:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
            
            if loss<=loss_thresold:
                print (f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias
    
customModel = insur_pred()
customModel.fit(x_train_scaled, y_train, epochs=8000, loss_thresold=0.4631)
y_predicted = customModel.predict(x_test_scaled)
print(y_predicted)