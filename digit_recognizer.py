from random import randint
from matplotlib import pyplot
from network import Network, Dataset, load, random_array
from utiles import load_data, test_recognizer, digits_dir
from os import path

#train = Dataset(*load_data("mnist_train.csv"))
test = Dataset(*load_data("mnist_test.csv"))

net = Network([784, 64, 10], 'L_ReLU', weight_range=(-0.1, 0.1), bias_range=(-0.1, 0.1))

#net = load(path.join(digits_dir, "recognizers/92.21.txt"))

#Settings
alpha = 0.03
beta = 0.3

#net.train_stochastic_momentum(train, alpha, beta, 60000, 10, True)
net.train_stochastic(test, alpha, 1000, 100, True)
#net.train_vanilla(test, alpha, 10, True)

accuracy, avg_cost = test_recognizer(net, test)
print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost {round(avg_cost, 3)}')

inp = ''
while inp != 'stop':
    inp = input()
    if inp == 'stop' or inp == '':
        continue
    if inp.split()[0] == 'train':
        if len(inp.split()) == 4:
            a, cycles, batch_size = inp.split()[1:]
            #net.train_stochastic(train, float(a), int(cycles), int(batch_size), True)
        else:
            print(f'Train takes 3 arguments, but {len(inp.split())-1} was given')
    elif inp == 'test':
        accuracy, avg_cost = test_recognizer(net, test)
        print(f'Accuracy: {round(100 * accuracy, 2)}% | Cost: {round(avg_cost, 3)}')
    elif inp.split()[0] == 'save':
        accuracy, avg_cost = test_recognizer(net, test)
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
            pyplot.imshow(test[rand].input_value.reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            net.process(test[rand].input_value)
            print(f'{net.layer_results[-1].argmax() == test[rand].output_value.argmax()} | Neuro: {net.layer_results[-1].argmax()} | Answer: {test[rand].output_value.argmax()} | Cost: {round(net.cost(test[rand].output_value), 3)} | ID: {rand}')
        pyplot.show(block = False)
    elif inp.isdigit():
        pyplot.imshow(test[int(inp)].input_value.reshape(28, 28), cmap=pyplot.get_cmap('gray'))
        net.process(test[int(inp)].input_value)
        print(f'{net.layer_results[-1].argmax() == test[int(inp)].output_value.argmax()} | Neuro: {net.layer_results[-1].argmax()} | Answer: {test[int(inp)].output_value.argmax()} | Cost: {round(net.cost(test[int(inp)].output_value), 3)} | Result: {net.layer_results[-1].round(3)}')
        pyplot.show(block = False)