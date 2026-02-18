from matplotlib import pyplot
import numpy as np
from network import Network, load, add_noise, apply_threshold, random_array
from time import perf_counter

def data_transform(num: int) -> list:
    transformed_data = [0 for _ in range(10)]
    transformed_data[num] = 1
    return(transformed_data)

def load_data(filename: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
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

def test(network: Network, dataset: list[np.ndarray], answerset: list[np.ndarray]) -> float:
    data_size = min(len(dataset), len(answerset))
    cost = 0
    for data, answer in zip(dataset, answerset):
        network.process(data)
        cost += network.cost(answer)
    return(cost / data_size)

def get_digit(network: Network, data: np.ndarray, noise_range: tuple[float, float], threshold: float) -> np.ndarray:
    return(apply_threshold(network.process(add_noise(data, noise_range)), threshold).reshape(28, 28))

train_x, train_y = load_data("mnist_train.csv")
test_x, test_y = load_data("mnist_test.csv")
avg_x, avg_y = load_data("avg_digits.txt")

#net = Network([10, 784], 'L_ReLU', 'sigmoid', factor_range=(-0.1, 0.1), bias_range=(-0.1, 0.1))
net = load("drawer41.8.txt")

#Settings
alpha = 0.1
beta = 0.4
noise_range = (-0.1, 0.1)
threshold = 0.3

net.train_stochastic_momentum(train_y, train_x, alpha, beta, 200, 60, True)

avg_cost = test(net, test_y, test_x)
print(f'Cost {round(avg_cost, 3)}')
net.save(f'drawer{round(avg_cost, 2)}.txt')

inp = ''
while inp != 'stop':
    inp = input()
    if inp != 'stop' and inp != '':
        if inp.split()[0] == 'train':
            if len(inp.split()) == 4:
                a, cycles, batch_size = inp.split()[1:]
                net.train_stochastic(train_y, train_x, float(a), int(cycles), int(batch_size), True)
            else:
                print(f'Train takes 3 arguments, but {len(inp.split())-1} was given')
        elif inp == 'test':
            avg_cost = test(net, test_y, test_x)
            print(f'Cost: {round(avg_cost, 3)}')
        elif inp.split()[0] == 'save':
            if len(inp) > 4:
                net.save(inp.split()[1])
                print(f'Saved {inp.split()[1]}')
            else:
                avg_cost = test(net, test_y, test_x)
                net.save(f'drawer{round(avg_cost, 3)}.txt')
                print(f'Saved drawer{round(avg_cost, 3)}')
        elif all(char in 'q' for char in inp):
            for _ in range(len(inp)):
                garbage = random_array(0, 1, 10)
                print(np.round(garbage, 2))
                pyplot.imshow(get_digit(net, garbage, noise_range, threshold), cmap=pyplot.get_cmap('gray'))
            pyplot.show(block = False)
        elif inp.isdigit and len(inp) == 1:
            pyplot.imshow(get_digit(net, np.array(data_transform(int(inp))), noise_range, threshold), cmap=pyplot.get_cmap('gray'))
            print(f'Digit: {test_y[int(inp)].argmax()} | Cost to average: {round(net.cost(avg_x[int(inp)]), 3)}')
            pyplot.show(block = False)