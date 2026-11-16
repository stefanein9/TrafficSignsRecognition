
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
from pathlib import Path


# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

MODEL_PATH = Path("models/best_model.keras")

IMAGE_SIZE = 32

# Названия 43 классов дорожных знаков
CLASS_NAMES = [
    "Ограничение скорости 20 км/ч",
    "Ограничение скорости 30 км/ч",
    "Ограничение скорости 50 км/ч",
    "Ограничение скорости 60 км/ч",
    "Ограничение скорости 70 км/ч",
    "Ограничение скорости 80 км/ч",
    "Конец ограничения скорости 80 км/ч",
    "Ограничение скорости 100 км/ч",
    "Ограничение скорости 120 км/ч",
    "Обгон запрещён",
    "Обгон грузовым автомобилям запрещён",
    "Главная дорога на следующем перекрёстке",
    "Главная дорога",
    "Уступите дорогу",
    "Движение без остановки запрещено",
    "Движение запрещено",
    "Движение грузовых автомобилей запрещено",
    "Въезд запрещён",
    "Прочие опасности",
    "Опасный поворот налево",
    "Опасный поворот направо",
    "Опасные повороты",
    "Неровная дорога",
    "Скользкая дорога",
    "Сужение дороги справа",
    "Дорожные работы",
    "Светофорное регулирование",
    "Пешеходный переход",
    "Дети",
    "Велосипедисты",
    "Гололёд или снег",
    "Дикие животные",
    "Конец всех ограничений",
    "Поворот направо",
    "Поворот налево",
    "Движение прямо",
    "Движение прямо или направо",
    "Движение прямо или налево",
    "Объезд препятствия справа",
    "Объезд препятствия слева",
    "Круговое движение",
    "Конец зоны запрета обгона",
    "Конец зоны запрета обгона грузовым автомобилям"
]


# ==========================================
# 2. ФУНКЦИЯ РАСПОЗНАВАНИЯ
# ==========================================

def predict_image(image_path):

    # Проверяем существование файла
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Файл не найден: {image_path}"
        )

    # Загружаем изображение
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            "Не удалось открыть изображение"
        )

    # Преобразуем BGR в RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Изменяем размер
    resized_image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Добавляем размерность пакета
    input_image = np.expand_dims(
        resized_image,
        axis=0
    )

    # Получаем вероятности классов
    probabilities = model.predict(
        input_image,
        verbose=0
    )[0]

    # Находим класс с максимальной вероятностью
    predicted_class = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_class]
    )

    # Выводим результат
    print("\nРЕЗУЛЬТАТ РАСПОЗНАВАНИЯ")
    print("=" * 40)

    print(
        "Номер класса:",
        predicted_class
    )

    print(
        "Название знака:",
        CLASS_NAMES[predicted_class]
    )

    print(
        f"Уверенность модели: {confidence * 100:.2f}%"
    )

    # Показываем три наиболее вероятных класса
    top_indices = np.argsort(probabilities)[-3:][::-1]

    print("\nТри наиболее вероятных варианта:")

    for class_id in top_indices:
        print(
            f"{CLASS_NAMES[class_id]}: "
            f"{probabilities[class_id] * 100:.2f}%"
        )

    # Отображаем изображение
    plt.figure(figsize=(5, 5))

    plt.imshow(image)

    plt.title(
        f"{CLASS_NAMES[predicted_class]}\n"
        f"Уверенность: {confidence * 100:.2f}%"
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================
# 3. ЗАГРУЗКА МОДЕЛИ
# ==========================================

print("Загрузка обученной нейросети...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Модель успешно загружена!")


# ==========================================
# 4. ЗАПУСК ПРОГРАММЫ
# ==========================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "Использование: "
            "python src/predict_image.py путь_к_изображению"
        )
        sys.exit(1)

    image_path = sys.argv[1]

    predict_image(image_path)
