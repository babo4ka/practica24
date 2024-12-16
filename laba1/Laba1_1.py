import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Функция для загрузки данных
def get_data():
    df = pd.read_csv("laba1.csv")
    x_df = df["ob"].to_numpy()
    y_df = df["ot"].to_numpy()
    return x_df, y_df


# Загрузка данных
x, y = get_data()

# Преобразуем данные в логарифмическую форму
X = np.log(x)
Y = np.log(y)

# Вычисляем коэффициенты методом наименьших квадратов
def get_coefficients(X, Y):
    n = len(X)
    B = (n * np.sum(X * Y) - np.sum(X) * np.sum(Y)) / (n * np.sum(X ** 2) - (np.sum(X)) ** 2)
    A = (np.sum(Y) - B * np.sum(X)) / n

    # Извлекаем коэффициенты степенной регрессии
    a = np.exp(A)  # возвращаем коэффициент a в первоначальной форме
    b = B          # коэффициент b

    return a, b


a, b = get_coefficients(X, Y)

# Вычисляем значения регрессии
y_pred = a * x ** b

# Выводим коэффициенты
print(f'Коэффициенты регрессии: a = {a:.2f}, b = {b:.2f}')

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


