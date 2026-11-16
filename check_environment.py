
import sys

print("Python version:", sys.version)

print("\nПроверка библиотек...")

try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print("TensorFlow ERROR:", e)

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except Exception as e:
    print("OpenCV ERROR:", e)

try:
    import numpy as np
    print("NumPy:", np.__version__)
except Exception as e:
    print("NumPy ERROR:", e)

try:
    import sklearn
    print("Scikit-learn:", sklearn.__version__)
except Exception as e:
    print("Scikit-learn ERROR:", e)

try:
    import matplotlib
    print("Matplotlib:", matplotlib.__version__)
except Exception as e:
    print("Matplotlib ERROR:", e)

try:
    from PIL import Image
    print("Pillow: OK")
except Exception as e:
    print("Pillow ERROR:", e)

try:
    import tkinter
    print("Tkinter: OK")
except Exception as e:
    print("Tkinter ERROR:", e)

print("\nПроверка завершена!")
