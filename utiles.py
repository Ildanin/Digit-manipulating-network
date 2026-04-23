import numpy as np
from time import perf_counter
from network import Network, Dataset
from os import path

digits_dir = path.dirname(__file__)

def data_transform(num: int) -> list:
    transformed_data = [0 for _ in range(10)]
    transformed_data[num] = 1
    return(transformed_data)

def load_data(filename: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    t1 = perf_counter()
    file_x: list[np.ndarray] = []
    file_y: list[np.ndarray] = []
    file = open(path.join(digits_dir, filename))
    for line in file:
        data = [int(x) for x in line.split(',')]
        file_x.append(np.array(data[1:]) / 255)
        file_y.append(np.array(data_transform(data[0])))
    file.close()
    t2 = perf_counter()
    print(f"{filename} loaded in {round(t2-t1, 3)} seconds")
    return(file_x, file_y)

def test_recognizer(network: Network, dataset: Dataset) -> tuple[float, float]:
    score = 0
    cost = 0
    for sample in dataset:
        if network.process(sample.input_value).argmax() == sample.output_value.argmax():
            score += 1
        cost += network.unaverage_cost(sample.output_value)
    return(score / len(dataset), cost / (len(dataset) * network.info[-1]))

def test_drawer(network: Network, dataset: Dataset) -> float:
    cost = 0
    for sample in dataset:
        network.process(sample.input_value)
        cost += network.unaverage_cost(sample.output_value)
    return(cost / (len(dataset) * network.info[-1]))