from math import *
import numpy as np
from random import randint
import os
from matplotlib import pyplot
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.datasets import mnist

class Network:
    def __init__(self,info,non_linear_func,non_linear_func_derivative,delta_factor=(-1,1),delta_bias=(-1,1),accuracy=2):
        self.info = info
        self.func = non_linear_func
        self.func_derivative = non_linear_func_derivative
        self.accuracy = accuracy
        self.layers = []
        self.layer_results = []
        for i in range(1,len(info)):
            self.layers.append(np.append(np.random.randint(delta_factor[0]*10**accuracy,delta_factor[1]*10**accuracy+1,(info[i],info[i-1])),(np.random.randint(delta_bias[0]*10**accuracy,delta_bias[1]*10**accuracy+1,(info[i],1))),axis=1)/(10**accuracy))

    def process(self,data_set):
        self.layer_results = []
        self.layer_results.append(data_set)
        for layer in self.layers:
            self.layer_results.append(np.array(list(map(self.func,layer.dot(np.append(self.layer_results[-1],1))))))
        return(self.layer_results[-1])

    def backpropagate(self,answer):
        gradient = []
        cumulative_difference = 2*(self.layer_results[-1]-answer)*np.array(list(map(self.func_derivative,self.layer_results[-1])))
        gradient.insert(0,np.reshape(cumulative_difference,[len(cumulative_difference),1])*np.append(self.layer_results[-2],1))
        for i in range(len(self.info)-3,-1,-1):
            cumulative_difference = self.layers[i+1][:,0:(self.info[i+1])].transpose().dot(cumulative_difference)*np.array(list(map(self.func_derivative,self.layer_results[i+1])))
            gradient.insert(0,np.reshape(cumulative_difference,[len(cumulative_difference),1])*np.append(self.layer_results[i],1))
        return(list(map(gradient_modify,gradient)))
    
    def mutate(self,gradient):
        for i, layer in enumerate(self.layers):
            layer += gradient[i]

    def cost(self,answer):
        return(sum((self.layer_results[-1]-answer)**2))

#constants
alpha = 0.004

def back_sum(back1, back2):
    b = []
    for i in range(len(back1)):
        b.append(back1[i]+back2[i])
    return(b)

def transform_answer(ans):
    t = np.zeros(10)
    t[ans] = 1 
    return(t)

def gradient_modify(a,coef=alpha):
    return(-coef*a)

def siqmoid(x):
    if x > 20:
        return(1)
    elif x < -20:
        return(0)
    else:
        return(1/(e**(-x)+1))

def siqmoid_derivative(x):
    if -15 < x < 15:
        return((e**x)/((e**x+1)**2))
    else:
        return(0)

def RELU(x):
    if x < 0:
        return(0)
    else:
        return(x)

def RELU_derivative(x):
    if x < 0:
        return(0)
    else:
        return(1)

def L_RELU(x):
    if x < 0:
        return(0.1*x)
    else:
        return(x)

def L_RELU_derivative(x):
    if x < 0:
        return(0.1)
    else:
        return(1)


net = Network([784, 100, 10], L_RELU, L_RELU_derivative, (-0.1, 0.1), (-10, 0))

(train_X, train_y), (test_X, test_y) = mnist.load_data()

n1 = 50000
n2 = 10000
inp = '0'

for _ in range(2):
    cost = 0
    right = 0
    for i in range(n1):
        if net.process(train_X[i].flatten()/255).argmax() == train_y[i]:
            right += 1
        cost += net.cost(transform_answer(train_y[i]))
        net.mutate(net.backpropagate(transform_answer(train_y[i])))
    alpha = alpha/5
    print(right/n1, cost/n1)

    
'''back = []
for i in range(1,len(net.info)):
    back.append(np.zeros([net.info[i],net.info[i-1]+1]))
while cost > 0.1 and counter != 2:
    cost = 0
    right = 0
    for i in range(6000):
        for j in range(10):
            if net.process(train_X[i*j].flatten()/255).argmax() == train_y[i*j]:
                right += 1
            back = back_sum(back,net.backpropagate(transform_answer(train_y[i*j]))) 
            cost += net.cost(transform_answer(train_y[i*j]))
        net.mutate(back)
        back = []
        for i in range(1,len(net.info)):
            back.append(np.zeros([net.info[i],net.info[i-1]+1]))
    print(right/n1, cost/n1)
    counter += 1'''


right = 0
for i in range(n2):
    net.process(test_X[i].flatten()/255)
    if net.layer_results[-1].argmax() == test_y[i]:
        right += 1
print(right/n2)

while inp != 'stop':
    inp = input()
    if inp != 'stop':
        if inp == 'q':
            garbage = np.random.randint(0,256,(28,28))
            pyplot.imshow(garbage, cmap=pyplot.get_cmap('gray'))
            print(net.process(garbage.flatten()/255),net.layer_results[-1].argmax())
            pyplot.show()
        elif inp == 'r':
            rand = randint(0, 10000)
            pyplot.imshow(test_X[rand], cmap=pyplot.get_cmap('gray'))
            print(rand,net.process(test_X[rand].flatten()/255),net.layer_results[-1].argmax(),test_y[rand])
            pyplot.show()
        else:
            pyplot.imshow(test_X[int(inp)], cmap=pyplot.get_cmap('gray'))
            print(int(inp),net.process(test_X[int(inp)].flatten()/255),net.layer_results[-1].argmax(),test_y[int(inp)])
            pyplot.show()