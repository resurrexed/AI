import torch 
import torch.nn as nn 
import numpy as np
import pandas as pd


# Создаем простую нейронную сеть
class NNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        nn.Module.__init__(self)
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size), # линейные сумматоры
            nn.Tanh(),                        # функция активации
            nn.Linear(hidden_size, out_size),
            nn.Sigmoid()                      # <-- исправлено: Tanh -> Sigmoid
        )

    def forward(self, X):
        pred = self.layers(X)
        return pred


# Загружаем данные
df = pd.read_csv('data.csv')

X = torch.Tensor(df.iloc[0:100, 0:2].values)

y = df.iloc[0:100, 2].values
y = torch.Tensor(y).view(-1, 1)  # преобразуем в вертикальный тензор


# Параметры сети
inputSize = X.shape[1]      # количество признаков
hiddenSizes = 3             # размер скрытого слоя
outputSize = 1              # один выходной нейрон для бинарной классификации

net = NNet(inputSize, hiddenSizes, outputSize)


# Выводим параметры сети
for name, param in net.named_parameters():
    print(name, param)


# Предсказание до обучения
with torch.no_grad():
    pred = net.forward(X)

# Приводим предсказания к классам: >= 0.5 → 1, < 0.5 → 0
pred_class = torch.Tensor((pred >= 0.5).float())

# Считаем ошибку
err = sum(abs(y - pred_class))
print('\nОшибка до обучения:', err.item())


# Функция потерь: BCELoss — правильнее для бинарной классификации
lossFn = nn.L1Loss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)

# Обучение
epohs = 10000
for i in range(epohs):
    pred = net.forward(X)
    loss = lossFn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f'Ошибка на {i+1} итерации: {loss.item():.4f}')


# Оценка после обучения
with torch.no_grad():
    pred = net.forward(X)

pred_class = (pred >= 0.5).float()
incorrect = torch.sum(y != pred_class).item()

print('\nОшибка (количество несовпавших ответов): ')
print(incorrect)
