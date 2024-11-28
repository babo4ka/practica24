import math
import random
import pandas as pd
import matplotlib.pyplot as plt


# вычисление эпсилон
def get_eps(h):
    return h/5

def linear_reg(a, b, x):
    return a+x*b


def linear_data(x, a, b):
    y = []
    eps = get_eps(math.trunc(x[-1] - x[0]))
    if eps < 0:
        eps = -eps

    eps = random.uniform(-eps, eps)
    for i in x:
        y.append(linear_reg(a, b, i) + eps)

    return y


def mnk(x, y):
    n = len(x)
    sumOfx2 = sum(math.pow(i, 2) for i in x)
    sumOfMuls = sum(x[i] * y[i] for i in range(n))

    a = (sum(y) * sumOfx2 - sum(x) * sumOfMuls) / (n * sumOfx2 - math.pow(sum(x), 2))
    b = (n * sumOfMuls - sum(y) * sum(x)) / (n * sumOfx2 - math.pow(sum(x), 2))

    return a, b


df = pd.read_csv("laba1.csv")
x = df["ob"].tolist()
y = df["ot"].tolist()


a, b = mnk(x, y)
y_preds = linear_data(x, a, b)

plt.ylabel("отклик")
plt.xlabel("объясняющая")
plt.scatter(x, y, label="данные")
plt.plot(x, y_preds, label="регрессия", c='r')
plt.legend()
plt.show()