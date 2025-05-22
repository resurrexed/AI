#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 08:12:48 2025

@author: stellar
"""
import random

numbers = []
for i in range(5):
    numbers.append(random.randint(1, 10))
print("Список чисел:", numbers)

sum_even = 0
for number in numbers:
    if number % 2 == 0:
        sum_even += number

print("Сумма чётных чисел:", sum_even)
