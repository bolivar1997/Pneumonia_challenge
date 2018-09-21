import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pydicom


# SOURCE_FOLDER_PATH = 'data/train_segmented/'
# NORM_FOLDER_PATH = 'data/train_segm_norm/'
# EQUALIZED_FOLDER_PATH = 'data/train_segm_equalized/'


SOURCE_FOLDER_PATH = 'data/whole_data/'
NORM_FOLDER_PATH = 'data/whole_data_norm/'
EQUALIZED_FOLDER_PATH = 'data/whole_data_equalized/'



def normalize_image(image):
    min_pixel = image[30:200, 30:200].min()
    max_pixel = image[30:200, 30:200].max()

    image = (image - min_pixel) * (255 / (max_pixel - min_pixel))

    return image


def equalize_image(image):
    equ = cv2.equalizeHist(image)
    return equ



def create_data_for_segmented():
    files = os.listdir(SOURCE_FOLDER_PATH)

    token = 0

    for file in files:
        if (token % 100 == 0):
            print(token)

        token += 1

        image = cv2.imread(SOURCE_FOLDER_PATH + '/' + file, 1)
        norm_image = normalize_image(image)
        equalized_image = equalize_image(image)
        cv2.imwrite(NORM_FOLDER_PATH + '/' + file, norm_image)
        cv2.imwrite(EQUALIZED_FOLDER_PATH + '/' + file, equalized_image)



def create_data_for_whole():
    token = 0
    files = os.listdir(SOURCE_FOLDER_PATH)

    for file in files:
        file_name = file.split('.')[0] + '.png'

        if (token % 100 == 0):
            print(token)

        token += 1

        ds = pydicom.dcmread(SOURCE_FOLDER_PATH + '/' + file)
        image_2d = ds.pixel_array.astype(np.uint8)

        norm_image = normalize_image(image_2d)
        equalized_image = equalize_image(image_2d)

        cv2.imwrite(EQUALIZED_FOLDER_PATH + '/' + file_name, equalized_image)
        #cv2.imwrite(EQUALIZED_FOLDER_PATH + '/' + file_name, _image)

        # Convert to float to avoid overflow or underflow losses.


if __name__ == '__main__':
    files = os.listdir(SOURCE_FOLDER_PATH)
    create_data_for_whole()
