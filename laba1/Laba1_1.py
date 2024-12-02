import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Исходные данные
df = pd.read_csv("laba1.csv")
x = df["ob"].to_numpy()
y = df["ot"].to_numpy()

# Преобразуем данные в логарифмическую форму
X = np.log(x)
Y = np.log(y)

# Вычисляем коэффициенты методом наименьших квадратов
n = len(x)
B = (n * np.sum(X * Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X**2) - (np.sum(X))**2)
A = (np.sum(Y) - B * np.sum(X)) / n

# Извлекаем коэффициенты степенной регрессии
a = np.exp(A)
b = B

# Вычисляем значения регрессии
y_pred = a * x**b

# Строим график
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Данные', s=20)
plt.plot(x, y_pred, color='red', label=f'Регрессия: y = {a:.2f} * x^{b:.2f}')
plt.xlabel('Объясняющая')
plt.ylabel('Отклик')
plt.title('Степенная регрессия')
plt.legend()
plt.grid(True)
plt.show()

# Выводим коэффициенты
print(f'Коэффициенты регрессии: a = {a:.2f}, b = {b:.2f}')