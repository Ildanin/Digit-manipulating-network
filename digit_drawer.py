from matplotlib import pyplot
import numpy as np
from network import Network, Dataset, load, add_noise, apply_threshold, random_array
from utiles import load_data, test_drawer, data_transform

def get_digit(network: Network, data: np.ndarray, noise_range: tuple[float, float], threshold: float) -> np.ndarray:
    return(apply_threshold(network.process(add_noise(data, noise_range)), threshold).reshape(28, 28))

train = Dataset(*reversed(load_data("mnist_train.csv")))
test = Dataset(*reversed(load_data("mnist_test.csv")))
avg_x, avg_y = load_data("avg_digits.txt")

#net = Network([10, 784], 'L_ReLU', 'sigmoid', factor_range=(-0.1, 0.1), bias_range=(-0.1, 0.1))
net = load("drawers/41.8.txt")

#Settings
alpha = 0.1
beta = 0.4
noise_range = (-0.1, 0.1)
threshold = 0.3

net.train_stochastic_momentum(train, alpha, beta, 200, 60, True)

avg_cost = test_drawer(net, test)
print(f'Cost {round(avg_cost, 3)}')
net.save(f"drawers/{round(100*avg_cost, 2)}.txt")

inp = ''
while inp != 'stop':
    inp = input()
    if inp == 'stop' or inp == '':
        continue
    if inp.split()[0] == 'train':
        if len(inp.split()) == 4:
            a, cycles, batch_size = inp.split()[1:]
            net.train_stochastic(train, float(a), int(cycles), int(batch_size), True)
        else:
            print(f'Train takes 3 arguments, but {len(inp.split())-1} was given')
    elif inp == 'test':
        avg_cost = test_drawer(net, test)
        print(f'Cost: {round(avg_cost, 3)}')
    elif inp.split()[0] == 'save':
        avg_cost = test_drawer(net, test)
        net.save(f'drawers/{round(avg_cost, 3)}.txt')
        print(f'Saved as {round(avg_cost, 3)} in drawers')
    elif all(char in 'q' for char in inp):
        for _ in range(len(inp)):
            garbage = random_array(0, 1, 10)
            print(np.round(garbage, 2))
            pyplot.imshow(get_digit(net, garbage, noise_range, threshold), cmap=pyplot.get_cmap('gray'))
        pyplot.show(block = False)
    elif inp.isdigit and len(inp) == 1:
        pyplot.imshow(get_digit(net, np.array(data_transform(int(inp))), noise_range, threshold), cmap=pyplot.get_cmap('gray'))
        print(f'Digit: {test[int(inp)].output_value.argmax()} | Cost to average: {round(net.cost(avg_x[int(inp)]), 3)}')
        pyplot.show(block = False)