import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
import pandas as pd

# Исходные данные
df = pd.read_csv("laba1.csv")
x = df["ob"].to_numpy()
y = df["ot"].to_numpy()


# Преобразуем данные в логарифмическую форму
X = np.log(x)
Y = np.log(y)

# Добавляем столбец единиц для вычисления intercept
X = sm.add_constant(X)

# Строим модель линейной регрессии
model = sm.OLS(Y, X).fit()

# Получаем коэффициенты
a = np.exp(model.params[0]) # Коэффициент a
b = model.params[1] # Коэффициент b

# Вычисляем предсказания
y_pred = np.exp(model.predict(X))

# Стандартная ошибка предсказания
pred_se = np.sqrt(model.mse_resid * (1/len(x) + (X[:, 1] - np.mean(X[:, 1]))**2 / np.sum((X[:, 1] - np.mean(X[:, 1]))**2)))

# Доверительный интервал для предсказаний
t_val = stats.t.ppf(1 - 0.025, len(x) - 2) # Критическое значение t
ci_upper = y_pred + t_val * pred_se
ci_lower = y_pred - t_val * pred_se

# Проверка на автокорреляцию (Durbin-Watson)
dw_stat = sm.stats.durbin_watson(model.resid)

# Проверка на гомоскедастичность (Breusch-Pagan)
bp_test = het_breuschpagan(model.resid, model.model.exog)

# Вывод результатов
print(f'Коэффициенты регрессии: a = {a:.2f}, b = {b:.2f}')
print(f'Статистика Дурбина-Уотсона: {dw_stat:.2f}')
print(f'Результат теста Бреуша-Пагана: p-значение = {bp_test[1]:.3f}')

# Построение графика
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Данные', s=20)
plt.plot(x, y_pred, color='red', label=f'Регрессия: y = {a:.2f} * x^{b:.2f}')
plt.fill_between(x, ci_lower, ci_upper, color='green', alpha=0.5, label='Доверительный коридор')
plt.xlabel('Объясняющая')
plt.ylabel('Отклик')
plt.title('Степенная регрессия с доверительным интервалом')
plt.legend()
plt.grid(True)
plt.show()

