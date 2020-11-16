import os
import glob
import numpy as np
from PIL import Image


def copy_dev_to_train(categories, path_dev, path_train):

    for cat in categories:

        remove_list = []
        path_cat = os.path.join(path_dev, cat)
        print("len dev {} = {}".format(cat, len(os.listdir(path_cat))))
        imgs_dev = [img for img in os.listdir(path_cat) if 'jpg' in img]

        for img_path in imgs_dev:

            img = Image.open(os.path.join(path_cat, img_path))
            img.save(os.path.join(path_train, cat, img_path), "JPEG")

            remove_list.append(os.path.join(path_cat, img_path))

        for remove_path in remove_list:
            os.remove(remove_path)

        print("New len dev {} = {}".format(cat, len(os.listdir(path_cat))))

    pass


def build_dev_from_train(categories, path_dev, path_train):

    for cat in categories:

        path_cat = os.path.join(path_train, cat)
        imgs_train = [img for img in os.listdir(path_cat) if 'jpg' in img]
        np.random.shuffle(imgs_train)

        n = len(imgs_train)
        print('{} -> {} train images'.format(cat, n))

        imgs_dev = imgs_train[:int(0.15 * n)]
        imgs_train = [img for img in imgs_train if img not in imgs_dev]

        print('{} -> {} new train images'.format(cat, len(imgs_train)))
        print('{} -> {} new dev images'.format(cat, len(imgs_dev)))

        path_dev_cat = os.path.join(path_dev, cat)

        for img_path in imgs_dev:

            img = Image.open(os.path.join(path_cat, img_path))
            img.save(os.path.join(path_dev_cat, img_path), format="JPEG")

            os.remove(os.path.join(path_cat, img_path))

    pass


if __name__ == '__main__':

    path_project = os.getcwd()
    path_birds = os.path.join(path_project, 'bird_dataset')
    path_train = os.path.join(path_birds, 'train_images')
    path_dev = os.path.join(path_birds, 'val_images')

    categories = [cat for cat in os.listdir(path_dev) if cat != '.DS_Store']

    copy_dev_to_train(categories, path_dev, path_train)
    build_dev_from_train(categories, path_dev, path_train)