import cv2
import numpy as np
import os
from PIL import Image


def image_resize(folder):
    """Takes in the location of the folder containing the pictures and loads all the pictures from the folder
    for pre-processing. Data pre-processing resizing the image to 200 X 200 pixels."""
    images = []
    num_images = 0
    location = folder+"_"
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            new_img = np.array(Image.fromarray(img).resize((200, 200), Image.ANTIALIAS))  # Resize the images to 50 X 50
            images.append(new_img)
            num_images += 1
            cv2.imwrite("{}/{}".format(location, filename), new_img)
    return None


image_resize("SLICY")
image_resize("2S1")
image_resize("BRDM-2")
image_resize("BTR-60")
image_resize("D7")
image_resize("T62")
image_resize("ZIL131")
image_resize("ZSU-23_4")
