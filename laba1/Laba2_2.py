import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance

# Данные
X = [3, 4, -4, -6, -1, 1, 0, 2]
Y = [-3, -6, -1, -5, 1, 2, 3, 4]
data = np.array(list(zip(X, Y)))


# Функция для инициализации центроидов
def initialize_centroids(data, k):
    centroids = [data[np.random.randint(0, len(data))]]
    for _ in range(1, k):
        distances = np.array([min([distance.euclidean(point, centroid) for centroid in centroids]) for point in data])
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, prob in enumerate(cumulative_probabilities):
            if r < prob:
                centroids.append(data[i])
                break
    return np.array(centroids)


# Функция для присвоения кластеров
def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = [distance.euclidean(point, centroid) for centroid in centroids]
        clusters.append(np.argmin(distances))
    return np.array(clusters)


# Функция для обновления центров кластеров
def update_centroids(data, labels, k):
    new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])
    return new_centroids


# Основная функция K-средних
def k_means(data, k, max_iters=100):
    centroids = initialize_centroids(data, k)
    for i in range(max_iters):
        labels = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, labels, k)

        # Проверка на сходимость
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    return centroids, labels


# Параметры
k = 3

# Запуск алгоритма K-средних
final_centroids, labels = k_means(data, k)

# Визуализация результата
plt.figure(figsize=(6, 4))
colors = ['r', 'g', 'b']
for i in range(k):
    cluster_points = data[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], c=colors[i], label=f'Кластер {i + 1}')

# Отображение центров кластеров
plt.scatter(final_centroids[:, 0], final_centroids[:, 1], c='black', marker='x', s=100, label='Центры кластеров')
plt.title('Результаты кластеризации методом K-средних')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.show()
