import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Путь к датасету
DATASET_PATH = Path("dataset/gtsrb")

# Загружаем таблицу с обучающими изображениями
train_df = pd.read_csv(DATASET_PATH / "Train.csv")

# Берём 5 случайных изображений
samples = train_df.sample(5, random_state=42)

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

for ax, (_, row) in zip(axes, samples.iterrows()):
    # Получаем путь к изображению
    image_path = DATASET_PATH / row["Path"]

    # Читаем изображение
    image = cv2.imread(str(image_path))

    if image is None:
        print("Не удалось открыть:", image_path)
        continue

    # OpenCV использует BGR, а Matplotlib — RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Показываем изображение
    ax.imshow(image)
    ax.set_title(f"Класс: {row['ClassId']}")
    ax.axis("off")

plt.tight_layout()
plt.show()
