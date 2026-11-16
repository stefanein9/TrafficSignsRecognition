
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

DATASET_PATH = Path("dataset/gtsrb")
MODEL_PATH = Path("models/best_model.keras")
RESULTS_PATH = Path("results")

IMAGE_SIZE = 32
NUM_CLASSES = 43


# ==========================================
# 2. ЗАГРУЗКА ТЕСТОВОЙ ВЫБОРКИ
# ==========================================

print("Загрузка Test.csv...")

test_df = pd.read_csv(DATASET_PATH / "Test.csv")

print("Количество тестовых изображений:", len(test_df))
print("Столбцы:", test_df.columns.tolist())

if "ClassId" not in test_df.columns or "Path" not in test_df.columns:
    raise ValueError(
        "В Test.csv должны присутствовать столбцы ClassId и Path"
    )


# ==========================================
# 3. ЗАГРУЗКА ИЗОБРАЖЕНИЙ
# ==========================================

images = []
labels = []

print("\nЗагрузка изображений...")

for index, row in test_df.iterrows():

    image_path = DATASET_PATH / row["Path"]

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Не удалось открыть изображение: {image_path}"
        )

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    images.append(image)
    labels.append(int(row["ClassId"]))

    if (index + 1) % 2000 == 0:
        print(f"Обработано: {index + 1}/{len(test_df)}")


X_test = np.array(images, dtype=np.uint8)
y_test = np.array(labels, dtype=np.int64)

print("\nРазмер тестовой выборки:", X_test.shape)


# ==========================================
# 4. ЗАГРУЗКА ОБУЧЕННОЙ МОДЕЛИ
# ==========================================

print("\nЗагрузка нейросети...")

model = tf.keras.models.load_model(MODEL_PATH)


# ==========================================
# 5. ОЦЕНКА МОДЕЛИ
# ==========================================

print("\nТестирование нейросети...")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\nРЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ")
print("=" * 40)

print(f"Точность: {test_accuracy * 100:.2f}%")
print(f"Ошибка: {test_loss:.4f}")


# ==========================================
# 6. ПОЛУЧЕНИЕ ПРЕДСКАЗАНИЙ
# ==========================================

print("\nПолучение предсказаний...")

predictions = model.predict(
    X_test,
    batch_size=64,
    verbose=1
)

y_pred = np.argmax(predictions, axis=1)


# ==========================================
# 7. ОТЧЁТ ПО КЛАССАМ
# ==========================================

print("\nОтчёт по классификации:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        zero_division=0
    )
)


# ==========================================
# 8. МАТРИЦА ОШИБОК
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(range(NUM_CLASSES))
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=list(range(NUM_CLASSES))
)

fig, ax = plt.subplots(figsize=(16, 16))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation="vertical",
    colorbar=False
)

plt.title("Матрица ошибок CNN")
plt.tight_layout()

RESULTS_PATH.mkdir(parents=True, exist_ok=True)

plt.savefig(
    RESULTS_PATH / "confusion_matrix.png",
    dpi=150
)

plt.show()

print("\nМатрица ошибок сохранена.")
print("Тестирование завершено!")
