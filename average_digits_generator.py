import numpy as np

def generate_avg_digits(filename: str) -> list[np.ndarray]:
    digit_counter = np.zeros(10)
    avg_digits = [np.zeros(784) for _ in range(10)]
    file = open(filename)
    for line in file:
        data = [int(x) for x in line.split(',')]
        digit_counter[data[0]] += 1
        avg_digits[data[0]] += np.array(data[1:])
    avg_digits = [np.round(avg_digits[i] / digit_counter[i]) for i in range(10)]
    file.close()
    return(avg_digits)

file = open("avg_digits.txt", 'w')
for i, digit in enumerate(generate_avg_digits("mnist_train.csv")):
    file.write(f'{i},{','.join([str(int(pix)) for pix in digit])}\n')