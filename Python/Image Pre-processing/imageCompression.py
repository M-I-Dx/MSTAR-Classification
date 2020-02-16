import cv2
import numpy as np
import os
from PIL import Image


def image_compression(folder):
    """Takes in the location of the folder containing the pictures and loads all the pictures from the folder
    for pre-processing. Data pre-processing includes converting the image to gray scale"""
    images = []
    num_images = 0
    location = folder+"_"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            images.append(img_grayscale)
            num_images += 1
            cv2.imwrite("{}/{}".format(location, filename), img_grayscale)
    return None


image_compression("SLICY")
image_compression("2S1")
image_compression("BRDM-2")
image_compression("BTR-60")
image_compression("D7")
image_compression("T62")
image_compression("ZIL131")
image_compression("ZSU-23_4")
