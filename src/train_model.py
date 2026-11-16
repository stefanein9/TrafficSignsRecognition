
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path


# ==========================================
# 1. НАСТРОЙКИ
# ==========================================

DATA_PATH = Path("dataset/processed/traffic_signs.npz")
MODEL_PATH = Path("models/traffic_signs_model.keras")
BEST_MODEL_PATH = Path("models/best_model.keras")
RESULTS_PATH = Path("results")

IMAGE_SIZE = 32
NUM_CLASSES = 43
BATCH_SIZE = 64
EPOCHS = 20


# ==========================================
# 2. ЗАГРУЗКА ДАННЫХ
# ==========================================

print("Загрузка подготовленного датасета...")

data = np.load(DATA_PATH)

X_train = data["X_train"]
X_val = data["X_val"]

y_train = data["y_train"]
y_val = data["y_val"]

print("Обучающая выборка:", X_train.shape)
print("Проверочная выборка:", X_val.shape)


# ==========================================
# 3. СОЗДАНИЕ CNN
# ==========================================

print("\nСоздание нейросети...")

model = tf.keras.Sequential([

    # Входное изображение
    tf.keras.layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    ),

    # Нормализация пикселей: 0-255 -> 0-1
    tf.keras.layers.Rescaling(1.0 / 255),

    # Первый свёрточный блок
    tf.keras.layers.Conv2D(
        32, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Второй свёрточный блок
    tf.keras.layers.Conv2D(
        64, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Третий свёрточный блок
    tf.keras.layers.Conv2D(
        128, (3, 3),
        activation="relu",
        padding="same"
    ),

    tf.keras.layers.MaxPooling2D((2, 2)),

    # Преобразование карт признаков в вектор
    tf.keras.layers.Flatten(),

    # Полносвязный слой
    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout защищает от переобучения
    tf.keras.layers.Dropout(0.5),

    # Выход: 43 класса
    tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ==========================================
# 4. КОМПИЛЯЦИЯ МОДЕЛИ
# ==========================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\nСтруктура нейросети:")
model.summary()


# ==========================================
# 5. ПОДГОТОВКА ОБУЧЕНИЯ
# ==========================================

MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.mkdir(parents=True, exist_ok=True)

callbacks = [

    # Сохраняем лучшую модель
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(BEST_MODEL_PATH),
        monitor="val_accuracy",
        save_best_only=True,
        mode="max"
    ),

    # Останавливаем обучение,
    # если качество перестало улучшаться
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),

    # Уменьшаем скорость обучения,
    # если качество перестало улучшаться
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=0.00001
    )
]


# ==========================================
# 6. ОБУЧЕНИЕ
# ==========================================

print("\nНачинаем обучение нейросети...")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)


# ==========================================
# 7. ОЦЕНКА РЕЗУЛЬТАТОВ
# ==========================================

print("\nОценка качества модели...")

val_loss, val_accuracy = model.evaluate(
    X_val,
    y_val,
    verbose=0
)

print(f"Точность на проверочной выборке: {val_accuracy * 100:.2f}%")
print(f"Функция потерь: {val_loss:.4f}")


# ==========================================
# 8. СОХРАНЕНИЕ МОДЕЛИ
# ==========================================

model.save(MODEL_PATH)

print("\nМодель сохранена:", MODEL_PATH)


# ==========================================
# 9. ПОСТРОЕНИЕ ГРАФИКОВ
# ==========================================

plt.figure(figsize=(12, 5))

# График точности
plt.subplot(1, 2, 1)

plt.plot(history.history["accuracy"], label="Обучение")
plt.plot(history.history["val_accuracy"], label="Проверка")

plt.title("Точность модели")
plt.xlabel("Эпоха")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()

# График функции потерь
plt.subplot(1, 2, 2)

plt.plot(history.history["loss"], label="Обучение")
plt.plot(history.history["val_loss"], label="Проверка")

plt.title("Функция потерь")
plt.xlabel("Эпоха")
plt.ylabel("Loss")
plt.legend()
plt.grid()

plt.tight_layout()

plt.savefig(
    RESULTS_PATH / "training_history.png",
    dpi=150
)

plt.show()

print("\nОбучение завершено!")
