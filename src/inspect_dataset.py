
import pandas as pd
from pathlib import Path

# Путь к датасету
DATASET_PATH = Path("dataset/gtsrb")

# Загружаем CSV
train_df = pd.read_csv(DATASET_PATH / "Train.csv")
test_df = pd.read_csv(DATASET_PATH / "Test.csv")
meta_df = pd.read_csv(DATASET_PATH / "Meta.csv")

print("=" * 50)
print("ИССЛЕДОВАНИЕ GTSRB")
print("=" * 50)

# 1. Размер обучающей выборки
print("\nОбучающая выборка:")
print("Количество изображений:", len(train_df))

# 2. Размер тестовой выборки
print("\nТестовая выборка:")
print("Количество изображений:", len(test_df))

# 3. Количество классов
print("\nКоличество классов:",
      train_df["ClassId"].nunique())

# 4. Распределение изображений по классам
print("\nРаспределение классов:")
print(train_df["ClassId"].value_counts().sort_index())

# 5. Структура CSV
print("\nСтолбцы Train.csv:")
print(train_df.columns.tolist())

print("\nПервые пять записей:")
print(train_df.head())

print("\nСтолбцы Meta.csv:")
print(meta_df.columns.tolist())

print("\nИсследование завершено!")
