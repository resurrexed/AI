import torch
import pandas as pd
import numpy as np

# Загружаем данные, указываем, что первая строка - заголовок
df = pd.read_csv('data.csv', header=0)

# Проверяем, что последний столбец называется '4'
print("Столбцы:", df.columns.tolist())

# Декодируем, если значения байтовые
df['4'] = df['4'].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)

# Преобразуем метки классов в числа
class_to_idx = {"Iris-setosa": 0, "Iris-versicolor": 1}
y = np.array([class_to_idx[cls] for cls in df['4']])

# Признаки — все столбцы кроме последнего
X = df.drop('4', axis=1).values

# Преобразуем в тензоры
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

print("\nРазмер входных данных:", X_tensor.shape)
print("Размер выходных данных:", y_tensor.shape)


#########################################################################
# Блок 2: Создание модели, функции потерь и оптимизатора
#########################################################################

print("\n=== Блок 2: Создание модели ===")

import torch.nn as nn

# Создаем модель: линейная регрессия (для многоклассовой классификации)
model = nn.Linear(4, 2)  # 4 входных признака -> 2 выходных класса

# Функция потерь: CrossEntropyLoss автоматически применяет Softmax
lossFn = nn.CrossEntropyLoss()

# Оптимизатор
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("Модель создана:")
print(model)


#########################################################################
# Блок 3: Обучение модели
#########################################################################

print("\n=== Блок 3: Обучение модели ===")

# Обучение по эпохам
for epoch in range(100):
    # Прямой проход
    outputs = model(X_tensor)
    loss = lossFn(outputs, y_tensor)

    # Обратный проход и шаг оптимизатора
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Эпоха [{epoch+1}/100], Ошибка: {loss.item():.4f}')


#########################################################################
# Блок 4: Оценка модели
#########################################################################

print("\n=== Блок 4: Оценка модели ===")

# Предсказания
with torch.no_grad():
    outputs = model(X_tensor)
    _, predicted = torch.max(outputs, 1)

# Сравниваем с истинными значениями
correct = (predicted == y_tensor).sum().item()
accuracy = correct / y_tensor.size(0)

print(f"\nТочность модели: {accuracy * 100:.2f}%")
