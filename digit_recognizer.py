from random import randint
from matplotlib import pyplot
import numpy as np
from network import Network, load, random_array
from time import perf_counter

def data_transform(num: int) -> list:
    transformed_data = [0 for _ in range(10)]
    transformed_data[num] = 1
    return(transformed_data)

def load_data(filename: str, ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    t1 = perf_counter()
    file_x: list[np.ndarray] = []
    file_y: list[np.ndarray] = []
    file = open(filename)
    for line in file:
        data = [int(x) for x in line.split(',')]
        file_x.append(np.array(data[1:]) / 255)
        file_y.append(np.array(data_transform(data[0])))
    file.close()
    t2 = perf_counter()
    print(f"{filename} loaded in {round(t2-t1, 3)} seconds")
    return(file_x, file_y)

def test(network: Network, dataset: list[np.ndarray], answerset: list[np.ndarray]) -> tuple[float, float]:
    data_size = min(len(dataset), len(answerset))
    score = 0
    cost = 0
    for data, answer in zip(dataset, answerset):
        if net.process(data).argmax() == answer.argmax():
            score += 1
        cost += network.cost(answer)
    return(score / data_size, cost / data_size)

train_x, train_y = load_data("mnist_train.csv")
test_x, test_y = load_data("mnist_test.csv")

#net = Network([784, 64, 10], 'L_ReLU', factor_range=(-0.1, 0.1), bias_range=(-0.1, 0.1))
net = load("recognizer96.71.txt")

#Settings
alpha = 0.04
beta = 0.3

net.train_stochastic_momentum(train_x, train_y, alpha, beta, 60000, 10, True)

accuracy, avg_cost = test(net, test_x, test_y)
print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost {round(avg_cost, 3)}')
net.save(f'recognizer{round(100 * accuracy, 2)}.txt')

inp = ''
while inp != 'stop':
    inp = input()
    if inp != 'stop' and inp != '':
        if inp.split()[0] == 'train':
            if len(inp.split()) == 4:
                a, cycles, batch_size = inp.split()[1:]
                net.train_stochastic(train_x, train_y, float(a), int(cycles), int(batch_size), True)
            else:
                print(f'Train takes 3 arguments, but {len(inp.split())-1} was given')
        elif inp == 'test':
            accuracy, avg_cost = test(net, test_x, test_y)
            print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost: {round(avg_cost, 3)}')
        elif inp.split()[0] == 'save':
            if len(inp) > 4:
                net.save(inp.split()[1])
            else:
                accuracy, avg_cost = test(net, test_x, test_y)
                net.save(f'recognizer{round(100 * accuracy, 2)}.txt')
        elif all(char in 'q' for char in inp):
            for _ in range(len(inp)):
                garbage = random_array(0, 1, (28, 28))
                pyplot.imshow(garbage, cmap=pyplot.get_cmap('gray'))
                print(net.process(garbage.flatten()), net.layer_results[-1].argmax())
            pyplot.show(block = False)
        elif all(char in 'r' for char in inp):
            for _ in range(len(inp)):
                rand = randint(0, 10000)
                pyplot.imshow(test_x[rand].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
                net.process(test_x[rand])
                print(f'{net.layer_results[-1].argmax() == test_y[rand].argmax()} | Neuro: {net.layer_results[-1].argmax()} | Answer: {test_y[rand].argmax()} | Cost: {round(net.cost(test_y[rand]), 3)} | ID: {rand}')
            pyplot.show(block = False)
        elif inp.isdigit():
            pyplot.imshow(test_x[int(inp)].reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            net.process(test_x[int(inp)])
            print(f'{net.layer_results[-1].argmax() == test_y[int(inp)].argmax()} | Neuro: {net.layer_results[-1].argmax()} | Answer: {test_y[int(inp)].argmax()} | Cost: {round(net.cost(test_y[int(inp)]), 3)} | Result: {net.layer_results[-1].round(3)}')
            pyplot.show(block = False)