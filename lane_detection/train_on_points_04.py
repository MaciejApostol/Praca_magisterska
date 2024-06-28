import os
import cv2
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.offline as po
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
    # Niepoprawnie zainicjowane urządzenie.
    pass

np.random.seed(10)


# Wizualizacja wyników uczenia
def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    fig = make_subplots(rows=2, cols=1, subplot_titles=('Accuracy', 'Loss'))
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['accuracy'], name='train_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_accuracy'], name='valid_accuracy',
                             mode='markers+lines'), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['loss'], name='train_loss',
                             mode='markers+lines'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hist['epoch'], y=hist['val_loss'], name='valid_loss',
                             mode='markers+lines'), row=2, col=1)

    fig.update_xaxes(title_text='Epoch', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_yaxes(title_text='Loss', row=2, col=1)
    fig.update_layout(width=1400, height=1000, title='Metrics')

    po.plot(fig, filename=os.path.join(filename, 'report.html'), auto_open=False)
    fig.write_image(os.path.join(filename, 'report.png'))


# Ładowanie danych
path = 'data'
dir_path = 'output'
if not os.path.exists(dir_path):
    os.mkdir(dir_path)

warp_data = pickle.load(open('data/data_array/160x80_warp_data.p', 'rb'))
warp_labels = pickle.load(open('data/data_array/160x80_warp_labels.p', 'rb'))
warp_coefficients = pickle.load(open('data/data_array/160x80_warp_coefficients.p', 'rb'))

data = pickle.load(open('data/data_array/160x80_data.p', 'rb'))
labels = pickle.load(open('data/data_array/160x80_labels.p', 'rb'))
coefficients = pickle.load(open('data/data_array/160x80_coefficients.p', 'rb'))

data_type = [warp_data, data]
labels_type = [warp_labels, labels]
coefficients_type = [warp_coefficients, coefficients]
fnames = ['train_1', 'train_2']

height = data[0].shape[0]
width = data[0].shape[1]
boundaries = [0, 0.6 * height]

# Parametry uczenia
epochs = 30
learning_rate = 0.001
batch_size = 32
input_shape = (height, width, 3)
loss = 'mse'
optimizer = 'adam'

for idx in range(2):
    output_path = os.path.join(dir_path, f'{fnames[idx]}')
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    logs_path = os.path.join(output_path, 'logs.txt')
    if os.path.exists(logs_path):
        os.remove(logs_path)

    csv_logger = CSVLogger(logs_path, append=True, separator='\t')

    model_path = os.path.join(output_path, 'model.h5')

    data = None
    data = np.array(data_type[idx])
    labels = np.array(labels_type[idx])

    # Podział danych na zbiór treningowy i walidacyjny
    shuffled_data, shuffled_labels = shuffle(data, labels)
    x_train, x_test, y_train, y_test = train_test_split(shuffled_data, shuffled_labels, test_size=0.2, random_state=10)

    # Generowanie danych treningowych i walidacyjnych
    train_generator = ImageDataGenerator()
    valid_generator = ImageDataGenerator()

    train_datagen = train_generator.flow(x_train, y_train, batch_size=batch_size)
    valid_datagen = valid_generator.flow(x_test, y_test, batch_size=batch_size)

    # Wizualizacja danych wejściowych
    from lane_detection_03 import visualise

    coefficients = coefficients_type[idx]
    start = boundaries[idx]
    y_range = np.linspace(start, height - 1, 3).astype(int)
    for i, (x, y) in enumerate(train_datagen):
        left_points = np.array(y[0][:3] * width).astype(int)
        right_points = np.array(y[0][3:] * width).astype(int)

        index = np.where(np.all(labels == y[0], axis=1))[0][0]

        left_curve = coefficients[index][:3]
        right_curve = coefficients[index][3:]

        image = visualise(x[0], left_curve, right_curve, start, show_lines=True)
        for j, y_ in enumerate(y_range):
            cv2.circle(image, (left_points[j][0], y_), 2, (0, 255, 0), -1)
            cv2.circle(image, (right_points[j][0], y_), 2, (0, 255, 0), -1)

        plt.figure(figsize=(4, 3))
        plt.imshow(image[:, :, ::-1])
        plt.title('Data image')
        plt.grid(False)
        plt.axis(False)

        plt.show(block=False)
        plt.pause(4)
        plt.close()

        break

    keras.backend.clear_session()

    # Tworzenie struktury modelu
    model = Sequential()
    model.add(BatchNormalization(input_shape=input_shape))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=6, activation='linear'))
    model.summary()

    # Kompilacja modelu
    model.compile(optimizer='adam',
                  loss=loss,
                  metrics=['accuracy'])

    # Uczenie modelu
    history = model.fit(x=train_datagen,
                        epochs=epochs,
                        validation_data=valid_datagen,
                        callbacks=csv_logger)

    # Zapisywanie wag
    model.save(model_path)
    plot_hist(history, filename=output_path)

    # Wyświetlanie metryk
    predictions = model.predict(valid_datagen)
    squeezed_test = np.squeeze(y_test)

    test_rmse = np.sqrt(model.evaluate(valid_datagen, verbose=0))[0]
    rmse = np.sqrt(model.evaluate(x_test, y_test, verbose=0))[0]

    print(f"Validation RMSE: {rmse}")

    r2 = r2_score(squeezed_test, predictions)
    print(f"Validation R^2 Score: {r2:.5f}")

    logs = open(logs_path, 'a')
    logs.write(f'\nepochs = {epochs}\n')
    logs.write(f'batch size = {batch_size}\n')
    logs.write(f'input shape = {input_shape}\n')
    logs.write(f'loss function = {loss}\n')
    logs.write(f'optimizer = {optimizer}\n')
    logs.write(f'rmse = {rmse}\n')
    logs.write(f'r2 = {r2}\n')
    logs.close()
