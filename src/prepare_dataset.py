
import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split


# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

DATASET_PATH = Path("dataset/gtsrb")
OUTPUT_PATH = Path("dataset/processed")

IMAGE_SIZE = 32
RANDOM_STATE = 42


# ==========================================
# 2. ЗАГРУЗКА CSV
# ==========================================

print("Загрузка таблицы Train.csv...")

train_df = pd.read_csv(DATASET_PATH / "Train.csv")

print("Количество изображений:", len(train_df))


# ==========================================
# 3. ЗАГРУЗКА ИЗОБРАЖЕНИЙ
# ==========================================

images = []
labels = []

print("\nНачинаем загрузку изображений...")

for index, row in train_df.iterrows():

    # Формируем полный путь к изображению
    image_path = DATASET_PATH / row["Path"]

    # Загружаем изображение
    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Не удалось открыть изображение: {image_path}"
        )

    # Переводим изображение из BGR в RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Изменяем размер до 32x32
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    images.append(image)

    # Записываем номер класса
    labels.append(row["ClassId"])

    # Показываем прогресс
    if (index + 1) % 5000 == 0:
        print(f"Обработано: {index + 1}/{len(train_df)}")


# Преобразуем списки в массивы NumPy
X = np.array(images, dtype=np.uint8)
y = np.array(labels, dtype=np.int64)


# ==========================================
# 4. РАЗДЕЛЕНИЕ ДАННЫХ
# ==========================================

print("\nРазделение данных...")

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# ==========================================
# 5. СОХРАНЕНИЕ
# ==========================================

OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

np.savez_compressed(
    OUTPUT_PATH / "traffic_signs.npz",
    X_train=X_train,
    X_val=X_val,
    y_train=y_train,
    y_val=y_val
)


# ==========================================
# 6. РЕЗУЛЬТАТЫ
# ==========================================

print("\n" + "=" * 45)
print("ПОДГОТОВКА ДАННЫХ ЗАВЕРШЕНА")
print("=" * 45)

print("Обучающие изображения:", len(X_train))
print("Проверочные изображения:", len(X_val))

print("Размер изображений:", X_train.shape[1:])
print("Количество классов:", len(np.unique(y)))

print("Минимальное значение пикселя:", X.min())
print("Максимальное значение пикселя:", X.max())

print("\nФайл сохранён:")
print(OUTPUT_PATH / "traffic_signs.npz")
