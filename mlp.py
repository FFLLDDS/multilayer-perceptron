import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

path_train = os.path.join(os.path.expanduser('~'), 'Downloads', 'zip.train')
path_test = os.path.join(os.path.expanduser('~'), 'Downloads', 'zip.test')

training_data = np.array(pd.read_csv(path_train, sep=' ', header=None))
test_data = np.array(pd.read_csv(path_test, sep =' ',header=None))

X_train, y_train = training_data[:,1:-1], training_data[:,0]
X_test, y_test = test_data[:,1:], test_data[:,0]


class Layer_Dense: 
    def __init__(self, n_inputs, n_neurons):
        self.has_weights = True
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output
        
    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        return np.dot(dvalues, self.weights.T)
        

class Activation_ReLU:
    def forward(self, inputs):
        self.has_weights = False
        self.inputs = inputs
        self.output = np.maximum(0, inputs.astype(float))
        return self.output
        
    def backward(self, dvalues):
        dReLU = np.clip(self.inputs, 0, np.inf)
        return dReLU * dvalues
    

class Activation_Sigmoid:
    def forward(self, inputs): 
        self.has_weights = False
        self.inputs = inputs
        self.output = 1./(1 + np.exp(-self.inputs))
        return self.output
    
    def backward(self, dvalues=1):
        dSigmoid = self.output * (1 - self.output)
        return dSigmoid * dvalues


def mse(pred, target, return_grad=True):
    if not return_grad:
        return np.mean((pred - target) ** 2)
    return np.mean((pred - target) ** 2), 2 * (pred - target)


class MLP: 
    def __init__(self, layers, num_classes):
        self.layers = layers
        self.num_classes = num_classes
        
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
            
    def backward(self, dvalues):
        for layer in reversed(self.layers):
            dvalues = layer.backward(dvalues)
    
    def gradient_descent(self, lr):
        for layer in self.layers:
            if layer.has_weights:
                layer.weights -= lr * layer.dweights
                layer.biases -= lr * layer.dbiases
    
    def fit(self, num_epochs, X_train, y_train, batch_size=32, lr=1e-3):
        num_datapoints = len(y_train) - len(y_train) % batch_size
        loss_training = []
        loss_test = []
        for epoch in tqdm(range(num_epochs)):
            loss_epoch = []
            for i in range(0, num_datapoints, batch_size):
                X = X_train[i:i+batch_size]
                y = np.eye(self.num_classes)[y_train[i:i+batch_size].astype(int)]
                pred = self.forward(X)
                loss, loss_grad = mse(pred, y)
                loss_epoch += [loss]
                self.backward(loss_grad)
                self.gradient_descent(lr)
            loss_training += [np.array(loss_epoch).mean()]
            loss_test += [mse(self.forward(X_test), np.eye(self.num_classes)[y_test.astype(int)],
                              return_grad=False)]
        return loss_training, loss_test
    
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    def accuracy(self, labels, predictions):
        return np.mean(labels == predictions)

      
model = MLP([
    Layer_Dense(16*16, 400),
    Activation_ReLU(),
    Layer_Dense(400, 10),
    Activation_Sigmoid()
], 10)

NUM_EPOCHS = 100

loss_training, loss_test = model.fit(NUM_EPOCHS, X_train, y_train)

pred = model.predict(X_test)
accu = model.accuracy(pred, y_test)
print(accu)

fig = plt.figure(figsize=(20, 10))
plt.plot(range(NUM_EPOCHS), loss_training, color='red', label='training_loss')
plt.plot(range(NUM_EPOCHS), loss_test, color='blue', label='test_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='upper right') # 'upper right' is default anyway
plt.show()


incorrectly_classified = np.argwhere((model.predict(X_test[:100])==y_test[:100])==False)

def show_numbers(X):
    fig = plt.figure(figsize=(20, 6))

    for i in range(len(X)):
        ax = plt.subplot(6, 15, i + 1)
        img = 255 - X[i].reshape((16, 16))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

show_numbers(X_test[incorrectly_classified])

def show_layer(W):
    fig = plt.figure(figsize=(20, 6))

    for i in range(len(W[0,:])):
        ax = plt.subplot(6, 15, i + 1)
        img = 255 - W[:,i].reshape((16, 16))
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()
  
show_layer(model.layers[0].weights[:,:90])