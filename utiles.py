import numpy as np
from time import perf_counter
from network import Network

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

def test_recognizer(network: Network, dataset: list[np.ndarray], answerset: list[np.ndarray]) -> tuple[float, float]:
    data_size = min(len(dataset), len(answerset))
    score = 0
    cost = 0
    for data, answer in zip(dataset, answerset):
        if network.process(data).argmax() == answer.argmax():
            score += 1
        cost += network.cost(answer)
    return(score / data_size, cost / data_size)

def test_drawer(network: Network, dataset: list[np.ndarray], answerset: list[np.ndarray]) -> float:
    data_size = min(len(dataset), len(answerset))
    cost = 0
    for data, answer in zip(dataset, answerset):
        network.process(data)
        cost += network.cost(answer)
    return(cost / data_size)