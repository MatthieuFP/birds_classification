# -*- coding: utf-8 -*-

"""
Created on Sat Nov 7 12:47:39 2020

@author: matthieufuteral-peter
"""

import os
import cv2
from PIL import Image
import pdb


def check_missing_bb(dataset, cat, missing):
    '''
    Check which images in bird dataset are not in the cropped dataset. In other words, check the images on which
    Mask R-CNN failed to detect any birds.

    Parameters:
        dataset (str) : 'train', 'test', 'val'
        cat (str) : bird species if dataset = 'train' or 'val' - 'mistery_category' if dataset = 'test'
        missing (dict) : dictionnary returned as output with updated missing bounding boxes

    Returns:
        dictionnary keys are 'train', 'test', 'val', values are dictionnaries with categories as keys and list of missing
        images as values.
    '''
    assert dataset in ['train', 'test', 'val'], "dataset must be 'train', 'test' or 'dev'. Unrecognized dataset"

    missing[dataset][cat] = []
    path_cat_cropped = os.path.join(path_cropped, '{}_cropped'.format(dataset), cat)
    path_cat_birds = os.path.join(path_birds, '{}_images'.format(dataset), cat)

    img_cropped = [img.split(r'.jpg')[0] for img in os.listdir(path_cat_cropped) if 'jpg' in img]
    img_cropped = [img[:-2] if r'_' in img[-2:] else img for img in img_cropped]

    img_birds = [img.split(r'.jpg')[0] for img in os.listdir(path_cat_birds) if 'jpg' in img]

    for img_bird in img_birds:
        if img_bird not in img_cropped:
            missing[dataset][cat].append(img_bird)

    return missing


def create_missing_bb(dataset, cat, missing):
    '''
    Given a dataset and a category, print the images and asked the user to draw a bounding box around the bird by
    clicking the upper left of the bird, press 'a' and then clicking the bottom right and press 'a'.
    Press 'c' once the latter steps were completed to print the next image. Save the images in
    './{dataset}_birds/{cat}'

    Parameters:
        dataset (str): 'train', 'test', 'val'
        cat (str): categories if 'train', 'val', 'mistery_category' if 'test'
        missing (dict): dict with path, cat and dataset of each image with no cropped bird

    Returns:
        None
    '''

    assert dataset in ['train', 'test', 'val'], "dataset must be 'train', 'test' or 'dev'. Unrecognized dataset"

    path_cat_cropped = os.path.join(path_cropped, '{}_cropped'.format(dataset), cat)
    path_cat_birds = os.path.join(path_birds, '{}_images'.format(dataset), cat)

    if not len(missing[dataset][cat]):
        print('No missing bounding box')

    else:
        print(len(missing[dataset][cat]))
        for fname in missing[dataset][cat]:
            path_bird = os.path.join(path_cat_birds, fname) + '.jpg'
            path_save_cropped = os.path.join(path_cat_cropped, fname) + '.jpg'

            bbox = []
            # Mouse function to select points on the image
            def select_point(event, x, y, flags, params):
                global point
                if event == cv2.EVENT_LBUTTONDOWN:
                    point = (x, y)

            cv2.namedWindow('frame')
            cv2.setMouseCallback('frame', select_point)

            image = cv2.imread(path_bird)

            while True:
                cv2.imshow('frame', image)
                k = cv2.waitKey(20)
                if k == ord('c'):  # c to continue
                    break
                elif k == ord('a'):
                    print(point)
                    bbox.append(point)

            assert len(bbox) == 2
            print(bbox)

            x0, y0, x1, y1 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[1][0]), int(bbox[1][1])
            cropped_image = image[y0:y1, x0:x1, ::-1]
            cropped_img = Image.fromarray(cropped_image)
            cropped_img.save(path_save_cropped)

    pass


if __name__ == '__main__':

    path_project = os.getcwd()
    path_birds = os.path.join(path_project, 'bird_dataset')
    path_cropped = os.path.join(path_project, 'cropped_birds')

    categories = [cat for cat in os.listdir(os.path.join(path_birds, 'train_images')) if cat != '.DS_Store']

    missing = {'train': {}, 'val': {}, 'test': {}}

    for cat in categories:
        missing = check_missing_bb('train', cat, missing)
        missing = check_missing_bb('val', cat, missing)
    missing = check_missing_bb('test', 'mistery_category', missing)

    pdb.set_trace()
    for cat in categories:
        create_missing_bb('train', cat, missing)
        create_missing_bb('val', cat, missing)
    create_missing_bb('test', 'mistery_category', missing)