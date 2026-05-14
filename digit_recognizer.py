from random import randint
from matplotlib import pyplot
from network import Network, Dataset, load, random_array
from utiles import load_data, test_recognizer, digits_dir
from os import path

#train = Dataset(*load_data("mnist_train.csv"))
test = Dataset(*load_data("mnist_test.csv"))

net = Network([("FC", 784, "L_ReLU"), 
               ("FC", 100, "L_ReLU"), 
               ("FC", 10, "softsign")], 
               weight_range=(-0.5, 0.5), bias_range=(-0.5, 0.5))

#net = load(path.join(digits_dir, "recognizers/97.62.txt"))

#Settings
alpha = 0.1
beta = 0.4

net.train_vanilla(test, alpha, 10, True)
#net.train_stochastic(test, alpha, 100, 1000, True)
#net.train_momentum(test, alpha, beta, 10, True)
#net.train_stochastic_momentum(test, alpha, beta, 100, 1000, True)

accuracy, avg_loss = test_recognizer(net, test)
print(f'Accuracy: {round(100 * accuracy, 2)}% | Loss {round(avg_loss, 3)}')

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
        accuracy, avg_loss = test_recognizer(net, test)
        print(f'Accuracy: {round(100 * accuracy, 2)}% | Loss: {round(avg_loss, 3)}')
    elif inp.split()[0] == 'save':
        accuracy, avg_loss = test_recognizer(net, test)
        net.save(f'recognizers/{round(100 * accuracy, 2)}.txt')
    elif all(char in 'q' for char in inp):
        for _ in range(len(inp)):
            garbage = random_array(0, 1, (28, 28))
            pyplot.imshow(garbage, cmap=pyplot.get_cmap('gray'))
            print(net.process(garbage.flatten()), net.last_result.argmax())
        pyplot.show(block = False)
    elif all(char in 'r' for char in inp):
        for _ in range(len(inp)):
            rand = randint(0, 10000)
            pyplot.imshow(test[rand].input_value.reshape(28, 28), cmap=pyplot.get_cmap('gray'))
            net.process(test[rand].input_value)
            print(f'{net.last_result.argmax() == test[rand].output_value.argmax()} | Neuro: {net.last_result.argmax()} | Answer: {test[rand].output_value.argmax()} | Loss: {round(net.loss(test[rand].output_value), 3)} | ID: {rand}')
        pyplot.show(block = False)
    elif inp.isdigit():
        pyplot.imshow(test[int(inp)].input_value.reshape(28, 28), cmap=pyplot.get_cmap('gray'))
        net.process(test[int(inp)].input_value)
        print(f'{net.last_result.argmax() == test[int(inp)].output_value.argmax()} | Neuro: {net.last_result.argmax()} | Answer: {test[int(inp)].output_value.argmax()} | Loss: {round(net.loss(test[int(inp)].output_value), 3)} | Result: {net.last_result.round(3)}')
        pyplot.show(block = False)