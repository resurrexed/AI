import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd

# Определяем архитектуру нейронной сети (оставляем без изменений)
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, out_size),
            nn.Tanh()
        )
    
    def forward(self, X):
        return self.layers(X)

# Загрузка и подготовка данных (адаптируем под новую задачу)
# Предполагаем, что данные находятся в файле 'customers.csv'
# Столбцы: возраст, доход, купит(да/нет)
df = pd.read_csv('data.csv')
X = torch.Tensor(df.iloc[:, 0:2].values)  # Признаки: возраст и доход (первые 2 столбца)
y = df.iloc[:, 2].values                  # Метки классов: купит/не купит (3й столбец)

y = torch.Tensor(np.where(y == "да", 1, -1).reshape(-1, 1))

inputSize = X.shape[1]    # 2 признака: возраст и доход
hiddenSizes = 3           # Количество нейронов в скрытом слое (оставляем 3)
outputSize = 1            # Один выходной нейрон для бинарной классификации

# Создаем экземпляр сети
net = NNet(inputSize, hiddenSizes, outputSize)

# Проверка работы сети до обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print("Ошибка до обучения:", err.item())

# Настройка обучения
lossFn = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

epochs = 100
for i in range(epochs):
    pred = net.forward(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item()}')

# Проверка работы сети после обучения
with torch.no_grad():
    pred = net.forward(X)

pred = torch.Tensor(np.where(pred >= 0, 1, -1).reshape(-1, 1))
err = sum(abs(y - pred)) / 2
print('\nОшибка после обучения:', err.item())

