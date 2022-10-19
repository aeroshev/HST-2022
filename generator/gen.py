import argparse

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int)
parser.add_argument('-o', '--output', type=str)
group = parser.add_mutually_exclusive_group()
group.add_argument('-b', '--bytes', type=str, nargs='+')
group.add_argument('-q', '--quantity', type=int)


def random_matrix(size: int, interval: int) -> np.ndarray:
    return np.random.uniform(low=-interval, high=interval, size=(size, size))


def calculate_q(memory: str, size: int) -> int:
    memory_names = [
        {"names": ["Ki", "KiB"], "mul": 1024},
        {"names": ["Mi", "MiB"], "mul": 1024 * 1024},
        {"names": ["Gi", "GiB"], "mul": 1024 * 1024 * 1024}
    ]
    value, name = memory.split(" ")
    value = int(value)
    name = name.strip()
    for mem in memory_names:
        if name in mem["names"]:
            value *= mem["mul"]
            break
    matrix_size = (size * size) * np.dtype(np.float64).itemsize
    return (value // matrix_size) + 1


if __name__ == '__main__':
    args = parser.parse_args()
    size = args.size
    quantity = args.quantity or calculate_q(' '.join(args.bytes), size)
    output_file = args.output or "matrixes.data"

    print(f"Quantity - {quantity}")
    print(f"Size - {size}")

    with open(output_file, "w") as file:
        file.write(f"{size} {quantity}\n")
        for _ in tqdm(range(quantity)):
            matrix = np.matrix(random_matrix(size, 100_000))
            for line in matrix:
                np.savetxt(file, line, fmt='%.4f')
