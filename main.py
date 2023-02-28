import os
import urllib.request
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras import layers

# Загрузка набора данных MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Предобработка данных
x_train = x_train.reshape((60000, 28, 28, 1)) / 255.0
x_test = x_test.reshape((10000, 28, 28, 1)) / 255.0

# Создание модели нейронной сети
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Загрузка модели
if os.path.exists('biom.h5'):
    model = keras.models.load_model('biom.h5')
else:
    print('Модель не найдена. Создание новой модели.')

# Обучение модели на 50 эпохах # В среднем прохождение одной валидации происходит за 1 цикл
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Сохранение модели в файл
model.save('biom.h5')

# Оценка точности модели
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nТочность модели на тестовых данных:', test_acc)

# Загрузка изображения и предобработка данных
img_url = input("Введите URL изображения: ")
response = urllib.request.urlopen(img_url)
img = Image.open(BytesIO(response.read()))
processed_image = np.array(img)
processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
processed_image = cv2.resize(processed_image, (28, 28))
processed_image = cv2.bitwise_not(processed_image)
processed_image = processed_image.reshape(1, 28, 28, 1) / 255.0

# Использование модели для распознавания объектов на изображении
prediction = model.predict(processed_image)
class_index = np.argmax(prediction)

# Определение наименования объекта
class_names = ['1', '2', '3', '4', '5', '6', '7']
if class_index >= len(class_names):
    object_name = "Неизвестный класс"
else:
    object_name = class_names[class_index]

# Вывод результата
print('Предсказание модели:', object_name)

# Отображение изображения
plt.imshow(processed_image.squeeze(), cmap='turbo')
plt.show()

# Визуализация результатов
plt.plot(history.history['accuracy'], label='Точность на обучающих данных')
plt.plot(history.history['val_accuracy'], label='Точность на тестовых данных')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()
