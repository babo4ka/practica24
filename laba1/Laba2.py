import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Исходные данные
X = [3,	4, -4, -6, -1, 1, 0, 2]
Y = [-3, -6, -1, -5, 1,	2, 3, 4]

# Объединяем данные в массив координат
data = np.array(list(zip(X, Y)))

# Функция для выбора начальных центроидов
def initialize_centroids(data, k):
    # Первый центроид выбирается случайно
    centroids = [data[np.random.randint(0, len(data))]]

    for _ in range(1, k):
        # Вычисляем расстояния от каждой точки до ближайшего центроида
        distances = np.array([min([distance.euclidean(point, centroid) for centroid in centroids]) for point in data])
        # Нормализуем расстояния для использования в качестве вероятностей
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()

        #Выбираем следующую точку на основе вероятности
        for i, prob in enumerate(cumulative_probabilities):
            if r < prob:
                centroids.append(data[i])
                break

    return np.array(centroids)

# Выбираем начальные центроиды
k = 3 # Количество кластеров
initial_centroids = initialize_centroids(data, k)

# Функция для присвоения кластеров
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        # Вычисляем расстояния до всех центроидов
        distances = [distance.euclidean(point, centroid) for centroid in centroids]
        # Присваиваем кластер с минимальным расстоянием
        clusters.append(np.argmin(distances))
    return np.array(clusters)

# Присваиваем точки кластерам
labels = assign_clusters(data, initial_centroids)

# Визуализация кластеров
plt.figure(figsize=(6, 4)) # Увеличиваем размер графика
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Кластер {i + 1}')

# Отображение центроидов
plt.scatter(initial_centroids[:, 0], initial_centroids[:, 1], s=200, c='yellow', marker='X', label='Центроиды')

# Расширяем границы осей
plt.xlim(-25, 10) # Устанавливаем диапазон для оси X
plt.ylim(-25, 10) # Устанавливаем диапазон для оси Y

# Настройка графика
plt.title('Распределение данных по кластерам (метод ближних соседей)')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()