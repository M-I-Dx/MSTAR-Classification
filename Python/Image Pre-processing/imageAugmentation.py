import imgaug.augmenters as iaa
import cv2
import os


augmentation = iaa.Sequential([iaa.Fliplr(0.5), iaa.Flipud(0.5)])


def image_augmentation(folder):
    images = []
    num_images = 0
    location = folder
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_aug = augmentation(images=img)
            images.append(img_aug)
            num_images += 1
            cv2.imwrite("{}/b{}".format(location, filename), img_aug)


image_augmentation("Image Augmentation Examples")
