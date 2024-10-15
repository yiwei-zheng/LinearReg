import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

class Point(object):
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        return

def linearReg(dataset: List[Point], eta: float = 0.2, steps: int = 1000) -> Tuple[float, float]:
    '''
       for y = ax + b, return tuple (a, b)
    '''
    a: float = 11
    b: float = 20

    x_tmp: List[float] = [p.x for p in dataset]
    y_tmp: List[float] = [p.y for p in dataset]
    lenth = len(x_tmp)
    x: np.ndarray = np.array(x_tmp).reshape((1, len(x_tmp)))
    y: np.ndarray = np.array(y_tmp).reshape((1, len(y_tmp)))

    def loss(a: float, b: float) -> float:
       return np.log(np.sum(np.square(y - a * x - b)) / lenth + 0.1)
    
    def biVarPartialDiff(func, a: float, b: float) -> Tuple[float, float]:
        delta = 1e-8
        val = func(a, b)
        da = (func(a + delta, b) - val) / delta
        db = (func(a, b + delta) - val) / delta
        return da, db
    
    for step in range(steps):
        print(f"loss = {loss(a, b)}")
        if abs(loss(a, b) - 0) <= 0.1:
            return a, b
        da, db = biVarPartialDiff(loss, a, b)
        a -= eta * da
        b -= eta * db
    
    return a, b


def main() -> int:
    p_set = [Point(1, 2), Point(2, 5), Point(3, 7), Point(4, 10)]
    a, b = linearReg(p_set)
    print(f"a: {a}")
    print(f"b: {b}")


    plt.scatter([p.x for p in p_set], [p.y for p in p_set])
    reg_x_val = np.linspace(0, 10, 100)
    reg_y_val = a * reg_x_val + b
    plt.plot(reg_x_val, reg_y_val)

    plt.show()
    return 0

if __name__ == "__main__":
    main()