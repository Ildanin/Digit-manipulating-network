from random import randint
from matplotlib import pyplot
from network import Network, load, random_array
from utiles import load_data, test_recognizer

train_x, train_y = load_data("mnist_train.csv")
test_x, test_y = load_data("mnist_test.csv")

#net = Network([784, 64, 10], 'L_ReLU', factor_range=(-0.1, 0.1), bias_range=(-0.1, 0.1))
net = load("recognizers/96.71.txt")

#Settings
alpha = 0.04
beta = 0.3

net.train_stochastic_momentum(train_x, train_y, alpha, beta, 60000, 10, True)

accuracy, avg_cost = test_recognizer(net, test_x, test_y)
print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost {round(avg_cost, 3)}')
net.save(f'recognizers/{round(100 * accuracy, 2)}.txt')

inp = ''
while inp != 'stop':
    inp = input()
    if inp == 'stop' and inp == '':
        continue
    if inp.split()[0] == 'train':
        if len(inp.split()) == 4:
            a, cycles, batch_size = inp.split()[1:]
            net.train_stochastic(train_x, train_y, float(a), int(cycles), int(batch_size), True)
        else:
            print(f'Train takes 3 arguments, but {len(inp.split())-1} was given')
    elif inp == 'test':
        accuracy, avg_cost = test_recognizer(net, test_x, test_y)
        print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost: {round(avg_cost, 3)}')
    elif inp.split()[0] == 'save':
        accuracy, avg_cost = test_recognizer(net, test_x, test_y)
        net.save(f'recognizers/{round(100 * accuracy, 2)}.txt')
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