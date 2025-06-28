import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt

# Проверка изображений на корректность
def verify_images(directory):
    print(f"Проверка изображений в {directory}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                path = os.path.join(root, file)
                try:
                    img = Image.open(path)
                    img.verify()  # Проверяет, что файл действительно является изображением
                    img.close()
                except Exception as e:
                    print(f"Ошибка в файле: {path} - {e}")
                    os.remove(path)
                    print("Файл удален.")

verify_images('./data/train')
verify_images('./data/test')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Трансформация данных
data_transforms = transforms.Compose([
    transforms.Resize(68),
    transforms.CenterCrop(64),
    transforms.ToTensor()
])

train_dataset = torchvision.datasets.ImageFolder(root='./data/train', transform=data_transforms)
test_dataset = torchvision.datasets.ImageFolder(root='./data/test', transform=data_transforms)

class_names = train_dataset.classes
print(f"Классы: {class_names}")

batch_size = 10
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Определение модели
class CnNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CnNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(8*8*64, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

num_classes = len(class_names)
net = CnNet(num_classes).to(device)

lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

# Обучение первой сети
import time
t = time.time()
num_epochs = 50
save_loss = []

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = lossFn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_loss.append(loss.item())
        if i % 100 == 0:
            print(f'Эпоха {epoch} из {num_epochs}, Шаг {i}, Ошибка: {loss.item()}')

print("Время обучения:", time.time() - t)

plt.figure()
plt.plot(save_loss)
plt.title("Loss во время обучения")
plt.show()

# Тестирование первой модели
correct_predictions = 0
num_test_samples = len(test_dataset)
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        pred = net(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print(f'Точность модели: {100 * correct_predictions / num_test_samples}%')

torch.save(net.state_dict(), 'CnNet_3classes.ckpt')

# Загрузка предобученной AlexNet
data_transforms_alex = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset_alex = torchvision.datasets.ImageFolder(root='./data/train', transform=data_transforms_alex)
test_dataset_alex = torchvision.datasets.ImageFolder(root='./data/test', transform=data_transforms_alex)

train_loader_alex = torch.utils.data.DataLoader(train_dataset_alex, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader_alex = torch.utils.data.DataLoader(test_dataset_alex, batch_size=batch_size, shuffle=False, num_workers=2)

net_alex = torchvision.models.alexnet(pretrained=True)

# Заморозка весов
for param in net_alex.parameters():
    param.requires_grad = False

# Изменение последнего слоя
new_classifier = net_alex.classifier[:-1]  # Убираем последний линейный слой
new_classifier.add_module('fc', nn.Linear(4096, num_classes))  # Добавляем новый
net_alex.classifier = new_classifier
net_alex = net_alex.to(device)

# Тестирование до обучения
correct_predictions = 0
num_test_samples = len(test_dataset_alex)
with torch.no_grad():
    for images, labels in test_loader_alex:
        images = images.to(device)
        labels = labels.to(device)
        pred = net_alex(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print(f'Точность модели до обучения: {100 * correct_predictions / num_test_samples}%')

# Обучение AlexNet
lossFn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net_alex.parameters(), lr=0.01)

t = time.time()
save_loss_alex = []

num_epochs = 10
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader_alex):
        images = images.to(device)
        labels = labels.to(device)
        outputs = net_alex(images)
        loss = lossFn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        save_loss_alex.append(loss.item())
        if i % 100 == 0:
            print(f'Эпоха {epoch} из {num_epochs}, Шаг {i}, Ошибка: {loss.item()}')

print("Время обучения AlexNet:", time.time() - t)

plt.figure()
plt.plot(save_loss_alex)
plt.title("Loss AlexNet")
plt.show()

# Тестирование после обучения
correct_predictions = 0
num_test_samples = len(test_dataset_alex)
with torch.no_grad():
    for images, labels in test_loader_alex:
        images = images.to(device)
        labels = labels.to(device)
        pred = net_alex(images)
        _, pred_class = torch.max(pred.data, 1)
        correct_predictions += (pred_class == labels).sum().item()

print(f'Точность модели после обучения: {100 * correct_predictions / num_test_samples}%')

# Визуализация предсказаний
inputs, classes = next(iter(test_loader_alex))
pred = net_alex(inputs.to(device))
_, pred_class = torch.max(pred.data, 1)

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

for i, j in zip(inputs, pred_class):
    img = i.cpu().numpy().transpose((1, 2, 0))  # Переводим в HxWxC
    img = std * img + mean
    img = np.clip(img, 0, 1)
    plt.imshow(img)
    plt.title(f"Предсказано: {class_names[j]}")
    plt.show(block=False)
    plt.pause(2)
    plt.clf()
