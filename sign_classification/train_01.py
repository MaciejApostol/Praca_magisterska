"""Plik z danymi potrzebnymi do przeprowadzenia klasyfikacji znaków pionowych można pobrać spod adresu:
    https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign?resource=download
    Pobrany plik archive.zip należy umieścić w folderze sign_classification."""

import os
import cv2
import zipfile
import warnings
import numpy as np
import pandas as pd
from imutils import paths
import plotly.offline as po
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split

import tensorflow as tf
from keras.models import Sequential
from keras.callbacks import CSVLogger
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense

warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

physical_devices = tf.config.list_physical_devices('GPU')
try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)
except:
    # Niepoprawnie zainicjowane urządzenie.
    pass


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


# Parametry uczenia
classes = 43
epochs = 15
learning_rate = 0.0001
batch_size = 32
input_shape = (30, 30, 3)
loss = 'categorical_crossentropy'
optimizer = 'adam'

# Ładowanie danych
path = 'data'
array_path = os.path.join(path, 'data_array')
output_path = os.path.join(path, 'output')
logs_path = os.path.join(output_path, 'logs.txt')

if not os.path.exists(path):
    with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
        print('Rozpakowywanie pliku')
        zip_ref.extractall('data')

if not os.path.exists(array_path):
    os.mkdir(array_path)

if not os.path.exists(output_path):
    os.mkdir(output_path)

if os.path.exists(logs_path):
    os.remove(logs_path)

csv_logger = CSVLogger(logs_path, append=True, separator='\t')

model_path = os.path.join(output_path, 'model.h5')

data_path = os.path.join(array_path, 'data.npy')
labels_path = os.path.join(array_path, 'labels.npy')

if not os.path.exists(data_path) or not os.path.exists(labels_path):
    data = []
    labels = []
    for class_ in range(classes):
        train_path = os.path.join(path, f'Train/{class_}')
        train_list = list(paths.list_images(train_path))

        for image_path in train_list:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (30, 30))
            data.append(image)
            labels.append(class_)

    data = np.array(data)
    labels = np.array(labels)
    np.save(data_path, data, allow_pickle=True, fix_imports=True)
    np.save(labels_path, labels, allow_pickle=True, fix_imports=True)

else:
    data = np.load(data_path)
    labels = np.load(labels_path)

# Podział danych na zbiór treningowy i walidacyjny
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
y_train = to_categorical(y_train, 43)
y_test = to_categorical(y_test, 43)

# Generowanie i augmentacja danych treningowych
train_generator = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Generowanie danych walidacyjnych
valid_generator = ImageDataGenerator()

train_datagen = train_generator.flow(x_train, y_train, batch_size=batch_size)
valid_datagen = valid_generator.flow(x_test, y_test, batch_size=batch_size)

# Tworzenie struktury modelu
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=x_train.shape[1:], activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=classes, activation='softmax'))
model.summary()

# Kompilacja modelu
model.compile(optimizer=optimizer,
              loss=loss,
              metrics=['accuracy'])

# Uczenie modelu
history = model.fit(x=x_train,
                    y=y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test))

# Zapisywanie wag
model.save(model_path)
plot_hist(history, filename=output_path)

logs = open(logs_path, 'a')
logs.write(f'\nepochs = {epochs}\n')
logs.write(f'batch size = {batch_size}\n')
logs.write(f'input shape = {input_shape}\n')
logs.write(f'loss function = {loss}\n')
logs.write(f'optimizer = {optimizer}\n')
logs.close()
