import cv2
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

path = 'data/camera_calibration/*.jpg'
images = glob.glob(path)

# Kryteria algorytmu takie jak: wymagana zmiana parametrów między iteracjami, maks. liczba iteracji
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Wymiary szachownicy – 9x6
grid = (9, 6)

# Przygotowanie współrzędnych 3D, z=0 – (0,0,0), (2,0,0), (2,0,0), ... (9,6,0)
objp = np.zeros((grid[1]*grid[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:grid[0], 0:grid[1]].T.reshape(-1, 2)

# Punkty na obiekcie w 3D
objpoints = []

# Punkty na zdjęciu w 2D
imgpoints = []

i = 0
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detekcja krawędzi pól szachownicy
    retval, corners = cv2.findChessboardCorners(gray, grid, None)

    if retval:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Lokacja punktów styku pól szachownicy
        corners2 = cv2.cornerSubPix(gray, corners, (15, 15), (-1, -1), criteria)

        # Wizualizacja punktów styku pól szachownicy
        cv2.drawChessboardCorners(img, grid, corners2, retval)
        cv2.imshow('camera_points', img)
        cv2.waitKey(500)

img = cv2.imread('data/camera_calibration/calibration1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Tworzenie macierzy kamery i współczynników dystorsji
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Nowa, optymalna macierz kamery
h, w = img.shape[:2]
newcameramatrix, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

dst = cv2.undistort(img, mtx, dist, None, mtx)
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

img_list = [img, dst]
name = ['Distortion', 'No distortion']

j = 0
for X, label in zip(img_list, name):
    j += 1
    plt.subplot(1, 2, j)
    plt.imshow(X)
    plt.title(label)
    plt.grid(False)
    plt.axis(False)

plt.show()

# Zapis macierzy kamery i współczynników dystorsji w celu ponownego użytku
pickle.dump(mtx, open(r'data/data_array/mtx.p', 'wb'))
pickle.dump(dist, open('data/data_array/dist.p', 'wb'))
