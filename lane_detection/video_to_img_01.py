"""Plik z danymi potrzebnymi do przeprowadzenia detekcji linii drogowych można pobrać spod adresu:
    https://drive.google.com/drive/folders/1iafmoSfAaO981a9yl5GxGOE7pS2ZO2Wa?usp=sharing
    Wystarczy pobrać plik.
    Po pobraniu pliku Video1.mp4 kod należy egzekwować od pozycji 01.
    Po pobraniu pliku data.zip kod należy egzekwować od pozycji 02.
    Niezależnie, który plik zostanie pobrany należy umieścić go w folderze lane_detection/data."""

import os
import cv2
import shutil
import numpy as np

# Ładowanie danych
path = 'data'
videos_path = path
train_path = os.path.join(path, 'train')
test_path = os.path.join(path, 'test')

# Tworzenie komunikatu początkowego
print('Delete previous data? [y/n]')
x = input()
x = x.lower()
if x != 'y' and x != 'n':
    raise Exception('Invalid input')

for folder in train_path, test_path:
    if os.path.exists(folder) and x == 'y':
        shutil.rmtree(folder)

    if not os.path.exists(folder):
        os.mkdir(folder)

# Ustawienia zapisu klatek
fps = 30
interval = 30
video = {"name": "Video1.mp4",
         "batch0": (15 * fps, 573 * fps),
         "batch1": (1815 * fps, 3805 * fps)}

values = list(video.values())[1:]
video_path = os.path.join(videos_path, video["name"])
cap = cv2.VideoCapture(video_path)

frames = 0
for value in values:
    diff = (value[1] - value[0]) / interval
    frames += diff

train = int(frames * 0.8)
test = int(frames - train)
i = 0
j = 0
k = 0

# Zapisywanie klatek
while cap.isOpened():
    _, image = cap.read()
    cropped_img = image[260:, :, :]

    if i < train:
        img_path = train_path + fr'\{i:05d}.jpg'
    else:
        img_path = test_path + fr'\{i:05d}.jpg'

    if values[k][0] < j <= values[k][1] and j % interval == 0:
        if np.any(image):
            if not os.path.exists(img_path):
                cv2.imwrite(img_path, cropped_img)
                print(f'{i} path: {img_path} – saving')
                i += 1
            else:
                print(f'{i} path: {img_path} – already exists')
                i += 1
        else:
            print('corrupted image')

    if j > values[k][1]:
        k += 1

    if j > values[-1][1]:
        print('end')
        break

    j += 1
