import os
import cv2
import pickle
import shutil
import zipfile
import numpy as np
from imutils import paths


# Funkcja ułatwiająca wizualizację zdjęć
def im_show(image, name='Image'):
    cv2.imshow(name, image)
    cv2.waitKey(0)


# Transformacja perspektywy
def warp_perspective(image, M):
    width = image.shape[1]
    height = image.shape[0]
    warp = cv2.warpPerspective(image, M, (width, height), flags=cv2.INTER_LINEAR)
    return warp


# Progowanie zdjęć
def threshold(image, T):
    _, image = cv2.threshold(image, T, 250, cv2.THRESH_BINARY)
    return image


# Tworzenie maski kolorów
def color_mask(img, thresh=(0, 255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray > thresh[0]) & (gray <= thresh[1])] = 255
    return binary_output


# Konwersja formatu BGR na skalę szarości
def gray_img(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Wizualizacja czworokąta źródłowego i docelowego wykorzystywanych do transformacji perspektywy.
def draw_lines(image, arr, point_color=(255, 0, 0), line_color=(0, 255, 0)):
    arr = arr.astype(int)
    copy = np.copy(image)
    for i in range(arr.shape[0]):
        x, y = arr[i][0], arr[i][1]
        x_0, y_0 = arr[i - 1][0], arr[i - 1][1]
        cv2.circle(copy, (x, y), radius=1, color=point_color, thickness=10)
        cv2.line(copy, (x, y), (x_0, y_0), color=line_color, thickness=4)
    return copy


# Przygotowanie zdjęć
def prepare(image, src, dst):
    undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    box = draw_lines(undistorted, src)
    box = draw_lines(box, dst, line_color=(0, 0, 255))
    warp = warp_perspective(undistorted, M)
    gray = gray_img(warp)
    max_val = max(np.amax(gray, axis=1)).astype(int)
    thresh = color_mask(warp, (max_val * 0.65, max_val))

    return warp, thresh


# Detekcja linii
def find_single_lane(side_current, count):
    side_left = side_current - margin
    side_right = side_current + margin
    cv2.rectangle(out_img, (side_left, low), (side_right, high), (0, 255, 0), 4)

    side_nonzero = ((nonzeroy >= low) & (nonzeroy <= high) &
                    (nonzerox >= side_left) & (nonzerox <= side_right)).nonzero()[0]

    side_indicator = True

    if (side_left < 0 or side_right > width) and len(side_nonzero) == 0:
        count += 1
    else:
        count = 0

    if count >= 17:
        side_indicator = False

    if len(side_nonzero) > minpix:
        side_current = int(np.mean(nonzerox[side_nonzero]))

    return side_current, side_nonzero, side_indicator, side_left, side_right


def find_lanes(image):
    global margin, minpix
    global out_img
    global low, high
    global nonzerox, nonzeroy
    global left_intercept, left_slope, right_intercept, right_slope
    global lefty, leftx, righty, rightx

    number = 35
    minpix = 50
    margin = 100

    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((image, image, image))

    midpoint = int(histogram.shape[0] // 2)
    left = np.argmax(histogram[:midpoint])
    right = midpoint + np.argmax(histogram[midpoint:])

    if np.argmax(histogram[:midpoint]) == 0:
        left = 0 + margin
    if np.argmax(histogram[midpoint:]) == 0:
        right = width - margin

    left_current = left
    right_current = right

    left_indicator = True
    right_indicator = True

    left_count = 0
    right_count = 0

    left_idx = []
    right_idx = []

    nonzero = np.nonzero(image)
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    win_height = int(height // number)

    for i in range(number):
        low = image.shape[0] - win_height * (i + 1)
        high = image.shape[0] - win_height * i

        if left_indicator and right_indicator:
            left_current, left_nonzero, left_indicator, _, left_right = find_single_lane(left_current, left_count)
            right_current, right_nonzero, right_indicator, right_left, _ = find_single_lane(right_current, right_count)
            left_idx.append(left_nonzero)
            right_idx.append(right_nonzero)

        elif left_indicator:
            left_current, left_nonzero, left_indicator, _, left_right = find_single_lane(left_current, left_count)
            left_idx.append(left_nonzero)

        elif right_indicator:
            right_current, right_nonzero, right_indicator, right_left, _ = find_single_lane(right_current, right_count)
            right_idx.append(right_nonzero)

        else:
            print('break')
            break

    try:
        left_idx = np.concatenate(left_idx)
        right_idx = np.concatenate(right_idx)
    except AttributeError:
        pass

    leftx1 = nonzerox[left_idx]
    lefty1 = nonzeroy[left_idx]
    rightx1 = nonzerox[right_idx]
    righty1 = nonzeroy[right_idx]

    if (len(leftx1) == 0 or len(leftx1) <= minpix) and (len(rightx1) == 0 or len(rightx1) <= minpix):
        print('no right and no left')
        leftx1 = previous_frame[0][0]
        lefty1 = previous_frame[0][1]
        rightx1 = previous_frame[0][2]
        righty1 = previous_frame[0][3]

    elif len(leftx1) <= minpix:
        print('no left')
        leftx1 = width - rightx1
        lefty1 = righty1

    elif len(rightx1) <= minpix:
        print('no right')
        rightx1 = width - leftx1
        righty1 = lefty1

    left_curve1 = np.polyfit(lefty1, leftx1, 2)
    right_curve1 = np.polyfit(righty1, rightx1, 2)

    left_nonzero1 = (
            (nonzerox > (left_curve1[0] * (nonzeroy ** 2) + left_curve1[1] * nonzeroy + left_curve1[2] - margin)) &
            (nonzerox < (left_curve1[0] * (nonzeroy ** 2) + left_curve1[1] * nonzeroy + left_curve1[2] + margin)))

    right_nonzero1 = (
            (nonzerox > (right_curve1[0] * (nonzeroy ** 2) + right_curve1[1] * nonzeroy + right_curve1[2] - margin)) &
            (nonzerox < (right_curve1[0] * (nonzeroy ** 2) + right_curve1[1] * nonzeroy + right_curve1[2] + margin)))

    leftx = nonzerox[left_nonzero1]
    lefty = nonzeroy[left_nonzero1]
    rightx = nonzerox[right_nonzero1]
    righty = nonzeroy[right_nonzero1]

    if len(leftx) <= minpix or len(rightx) <= minpix:
        leftx, lefty, rightx, righty = leftx1, lefty1, rightx1, righty1

    return leftx, lefty, rightx, righty, out_img


# Dopasowanie wielomianów drugiego stopnia
def fit_poly(leftx, lefty, rightx, righty):
    left_curve = np.polyfit(lefty, leftx, 2)
    right_curve = np.polyfit(righty, rightx, 2)

    return left_curve, right_curve


# Generowanie punktów leżących na wielomianach
def generate_points(image, left_curve, right_curve, start=0, stop=0, num=16, labels=False):
    width = image.shape[1]
    height = image.shape[0]

    if stop:
        end = stop
    else:
        end = height - 1

    y = np.linspace(start, end, num).astype(int).reshape((-1, 1))
    fit_left = left_curve[0] * y ** 2 + left_curve[1] * y + left_curve[2]
    fit_right = right_curve[0] * y ** 2 + right_curve[1] * y + right_curve[2]

    if labels:
        labels_points = np.concatenate((fit_left, fit_right)) / width
        return y, labels_points

    empty = []
    flipud = False

    for arr in fit_left, fit_right:
        arr = arr.astype(int)
        con = np.concatenate((arr, y), axis=1)

        if flipud:
            con = np.flipud(con)

        flipud = True
        empty.append(con)

    visualise_points = np.array(empty)

    return visualise_points


# Skalowanie wielomianów i obliczanie ich na zdjęciach ze zmienioną perspektywą
def scale_and_unwarp(image, left_curve, right_curve, unwarp=False):
    points_arr = generate_points(image, left_curve, right_curve)

    nonzero = []
    for arr in points_arr:
        side = np.zeros((height, width))
        side = cv2.polylines(side, [arr], isClosed=False, color=1, thickness=20)
        if unwarp:
            side = cv2.warpPerspective(side, M_inv, (width, height), flags=cv2.INTER_LINEAR)

        side = cv2.resize(side, (s_width, s_height))

        nonzerox = side.nonzero()[1]
        nonzeroy = side.nonzero()[0]
        nonzero.append(nonzerox)
        nonzero.append(nonzeroy)

    leftx, lefty, rightx, righty = nonzero

    if len(leftx) == 0:
        leftx = width - rightx
        lefty = righty

    if len(rightx) == 0:
        rightx = width - leftx
        righty = lefty

    return leftx, lefty, rightx, righty


# Wizualizacja wielomianów i leżących na nich punktach
def visualise(image, left_curve, right_curve, start=0, stop=0, show_lines=True, show_points=False):
    points_arr = generate_points(image, left_curve, right_curve, start, stop)
    copy = np.copy(image)

    for idx, arr in enumerate(points_arr):
        if show_lines:
            cv2.polylines(copy, [arr], isClosed=False, color=(255, 0, 0), thickness=4)

        if show_points:
            for point in arr:
                cv2.circle(copy, tuple(point), 5, (255, 0, 0), -1)

    return copy


# Wizualizacja masek
def visualise_masks(image, left_curve, right_curve, start=0, stop=0, line_label=False):
    poly = np.zeros_like(image)
    width = poly.shape[1]

    points_arr = generate_points(image, left_curve, right_curve, start, stop)
    colors = [0, 0, 0]

    if line_label:
        channel = 0
        colors[channel] = 255
        for arr in points_arr:
            l_offset = np.copy(arr)
            r_offset = np.copy(arr)
            l_offset[:, 0] += width // 128
            r_offset[:, 0] -= width // 128
            points = np.vstack((l_offset, np.flipud(r_offset)))
            poly = cv2.fillPoly(poly, [points], colors)

    else:
        channel = 1
        colors[channel] = 255
        points = np.vstack((points_arr[0], points_arr[1]))
        poly = cv2.fillPoly(poly, [points], colors)

    out_frame = cv2.addWeighted(image, 1, poly, 0.5, 0)
    poly = poly[:, :, channel]

    return poly, out_frame


# Parametry transformacji perspektywy
def params():
    template = np.float32([[290, 410*4/3], [550, 285*4/3]])
    src = np.float32([[template[0][0], template[0][1]],
                      [template[1][0], template[1][1]],
                      [width - template[1][0], template[1][1]],
                      [width - template[0][0], template[0][1]]])

    dst = np.float32([[0, height],
                      [0, 0],
                      [width, 0],
                      [width, height]])

    return src, dst


# Tworzenie komunikatu początkowego
def make_input(message):
    print(message, ' [y/n]')
    x = input()
    x = x.lower()
    if x != 'y' and x != 'n':
        raise Exception('Invalid input')

    return x


# Przygotowanie danych potrzebnych i inicjalizacji utworzonych funkcji
def main(path):
    global width, height
    global s_width, s_height
    global previous_frame
    global M, M_inv
    global mtx, dist

    data_path = os.path.join(path, 'train')
    array_path = os.path.join(path, 'data_array')
    frames_path1 = os.path.join(path, 'frames1')
    frames_path2 = os.path.join(path, 'frames2')
    labels_path1 = os.path.join(path, 'labels1')
    labels_path2 = os.path.join(path, 'labels2')

    if os.path.exists(data_path):
        if len(os.listdir(data_path)) == 0:
            shutil.rmtree(data_path)

    if not os.path.exists(data_path):
        unzip_path = 'unzipped_data'
        with zipfile.ZipFile('data/data.zip', 'r') as zip_ref:
            print('Unzipping file')
            zip_ref.extractall(unzip_path)

        for file in os.listdir(unzip_path):
            shutil.move(os.path.join(unzip_path, file), os.path.join(path, file))
        shutil.rmtree(unzip_path)

    if not os.path.exists(array_path):
        os.mkdir(array_path)

    x = make_input('Delete previous data?')
    for folder_path in frames_path1, frames_path2, labels_path1, labels_path2:
        if os.path.exists(folder_path) and x == 'y':
            shutil.rmtree(folder_path)

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)

    data_list = list(paths.list_images(data_path))
    image = cv2.imread(data_list[0])
    width = image.shape[1]
    height = width // 2
    scale_factor = 1 / 8
    s_width = int(width * scale_factor)
    s_height = int(height * scale_factor)

    data = []
    warp_data = []
    img_labels1 = []
    img_labels2 = []
    labels = []
    warp_labels = []
    coefficients = []
    warp_coefficients = []

    src, dst = params()

    mtx = pickle.load(open('data/data_array/mtx.p', 'rb'))
    dist = pickle.load(open('data/data_array/dist.p', 'rb'))

    M = cv2.getPerspectiveTransform(src, dst)
    M_inv = cv2.getPerspectiveTransform(dst, src)

    M_path = os.path.join(array_path, f'M_Video1.npy')
    M_inv_path = os.path.join(array_path, f'M_inv_Video1.npy')

    np.save(M_path, M)
    np.save(M_inv_path, M_inv)

    i = 0
    for image_path in data_list:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
        warp, img = prepare(image, src, dst)

        leftx0, lefty0, rightx0, righty0, out_img = find_lanes(img)

        previous_frame = [leftx0, lefty0, rightx0, righty0]

        left_curve0, right_curve0 = fit_poly(leftx0, lefty0, rightx0, righty0)

        if scale_factor == 1:
            leftx, lefty, rightx, righty = leftx0, lefty0, rightx0, righty0
        else:
            leftx, lefty, rightx, righty = scale_and_unwarp(image, left_curve0, right_curve0, unwarp=False)

        t_leftx, t_lefty, t_rightx, t_righty = scale_and_unwarp(image, left_curve0, right_curve0, unwarp=True)

        left_curve, right_curve = fit_poly(leftx, lefty, rightx, righty)
        t_left_curve, t_right_curve = fit_poly(t_leftx, t_lefty, t_rightx, t_righty)

        curves = np.concatenate((left_curve, right_curve))
        t_curves = np.concatenate((t_left_curve, t_right_curve))

        start = int(s_height * 0.6)
        stop = scale_factor * src[0][1]
        frame = cv2.resize(image, (s_width, s_height))
        poly1, out_frame1 = visualise_masks(frame, t_left_curve, t_right_curve, start, stop)
        poly2, out_frame2 = visualise_masks(frame, t_left_curve, t_right_curve, start, stop, True)
        image = cv2.resize(image, (s_width, s_height)) / 255
        warp = cv2.resize(warp, (s_width, s_height)) / 255

        y, curves_points = generate_points(warp, left_curve, right_curve, num=3, labels=True)
        y_t, t_curves_points = generate_points(image, t_left_curve, t_right_curve, start, num=3, labels=True)

        visualization = visualise(image, t_left_curve, t_right_curve, start)

        points_visualization = np.copy(image)
        for k, y_ in enumerate(y_t):
            points_visualization = cv2.circle(points_visualization, (int(t_curves_points[k] * s_width), y_[0]), 4,
                                              (0, 255, 0), -1)
            points_visualization = cv2.circle(points_visualization, (int(t_curves_points[k + 3] * s_width), y_[0]), 4,
                                              (0, 255, 0), -1)

        # visualise_list = [visualization, points_visualization, out_frame1, out_frame2]
        # for img_vis in visualise_list:
        #     im_show(img_vis)

        save_frame1 = frames_path1 + fr'\{os.path.basename(image_path)}'
        save_frame2 = frames_path2 + fr'\{os.path.basename(image_path)}'
        save_label1 = labels_path1 + fr'\{os.path.basename(image_path)}'
        save_label2 = labels_path2 + fr'\{os.path.basename(image_path)}'

        save_list = [os.path.exists(save_path) for save_path in [save_frame1, save_frame2, save_label1, save_label2]]
        if False in save_list:
            cv2.imwrite(save_frame1, out_frame1)
            cv2.imwrite(save_frame2, out_frame2)
            cv2.imwrite(save_label1, poly1)
            cv2.imwrite(save_label2, poly2)
            print(f'{i} path: {image_path} – saving frames and labels')
        else:
            print(f'{i} path: {image_path} – already processed')

        poly1 = poly1 / 255
        poly2 = poly2 / 255

        image = image.astype('float32')
        warp = warp.astype('float32')
        poly1 = poly1.astype('float32')
        poly2 = poly2.astype('float32')

        data.append(image)
        warp_data.append(warp)
        img_labels1.append(poly1)
        img_labels2.append(poly2)

        labels.append(t_curves_points)
        warp_labels.append(curves_points)
        coefficients.append(t_curves)
        warp_coefficients.append(curves)

        i += 1

    print('end')

    pickle.dump(data, open(f'data/data_array/{s_width}x{s_height}_data.p', 'wb'))
    pickle.dump(warp_data, open(f'data/data_array/{s_width}x{s_height}_warp_data.p', 'wb'))

    pickle.dump(img_labels1, open(f'data/data_array/{s_width}x{s_height}_img_labels1.p', 'wb'))
    pickle.dump(img_labels2, open(f'data/data_array/{s_width}x{s_height}_img_labels2.p', 'wb'))
    pickle.dump(labels, open(f'data/data_array/{s_width}x{s_height}_labels.p', 'wb'))
    pickle.dump(warp_labels, open(f'data/data_array/{s_width}x{s_height}_warp_labels.p', 'wb'))

    pickle.dump(coefficients, open(f'data/data_array/{s_width}x{s_height}_coefficients.p', 'wb'))
    pickle.dump(warp_coefficients, open(f'data/data_array/{s_width}x{s_height}_warp_coefficients.p', 'wb'))


path = 'data'
if __name__ == "__main__":
    main(path)
