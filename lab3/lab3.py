import torch 

# 1. Создаём случайный тензор
drt = torch.randint(low=1, high=10, size=(3, 3))
print("Случайный тензор drt:")
print(drt)

# 2. Преобразуем в float32 и включаем градиенты
drt = drt.to(dtype=torch.float32)
drt.requires_grad = True

# 3. Присваиваем rt = drt
rt = drt
print("\nrt = drt:")
print(rt)

# 4. Возводим в квадрат
rt = rt ** 2
print("\nrt^2:")
print(rt)

# 5. Умножаем на случайное число (скаляр)
scalar = torch.randint(low=1, high=10, size=(1, 1)).float()
rt = rt * scalar
print(f"\nУмножение на скаляр {scalar.item()}:")
print(rt)

# 6. Берём экспоненту
rt = torch.exp(rt)
print("\nexp(rt):")
print(rt)

# 7. Вычисляем градиенты — используем .sum(), чтобы получить скаляр
rt.sum().backward()

# 8. Выводим производную d(rt)/d(drt)
print("\nПроизводная d(rt)/d(drt):")
print(drt.grad)

