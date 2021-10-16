#!/usr/bin/env python3
import os
import glob
import numpy as np
import cv2
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
from keras import backend as K
import pickle

MODULE_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
SAVE_PATH_DIR = os.path.join(MODULE_DIR_PATH, "database")
if not os.path.exists(SAVE_PATH_DIR):
    os.makedirs(SAVE_PATH_DIR)
IMAGES_PATH = os.path.join(MODULE_DIR_PATH, "images")
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)
DATABASE_PATH = os.path.join(SAVE_PATH_DIR, "database.pickle")

K.set_image_data_format("channels_first")   # tensorflow format e.g. (2, 96, 96)

FRmodel = faceRecoModel(input_shape=(3, 96, 96))


def triplet_loss(y_pred, alpha=0.3):
    """
    Calculate loss for back propagation
    Finds the difference between the anchor (true) image and the positive image
    Finds the difference between the anchor (true) image and the negative image
    This loss is then used to minimize the difference between positive and anchor image
    while maximizing difference between the anchor and negative image
    :param y_pred: (anchor, positive, negative)
    :param alpha: constant used to ensure model does not set all weights to 0 (default =0.3)
    :return: loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]

    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))

    return loss


FRmodel.compile(optimizer="adam", loss=triplet_loss, metrics=["accuracy"])
load_weights_from_FaceNet(FRmodel)


def preprocess_image(image):
    # resized_image = cv2.resize(image, (96, 96))
    processed_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # grayscale
    return processed_img


def prepare_database(use_avg=False):
    """
    Scan the images directory and find all folders. In each found folder take the image(s)
    found and generate a 1X128 tensor encoding and save it to the in-memory database
    When multiple images are found in 1 folder, all are encoded and the encodings saved.

    :param use_avg: Specifies whether the all encodings should be saved for a given folder(/identity) or if
    the multiple encodings should be averaged and only a single tensor of average values saved (default False).
    :return: {"name": [tensor1...tensor_n], ...} tensor = 1x128
    """
    database = {}

    for name in os.listdir(IMAGES_PATH):  # look for all folders in images dir (except 'ignore' folder)
        person_folder = os.path.join(IMAGES_PATH, name)
        # print(f"{name} = {person_folder}")
        if os.path.isdir(person_folder):
            if name.lower() == "ignore":  # skip the 'ignore' folder
                continue
            folder_content = os.listdir(person_folder)  # get all files in the folder representing a person
            encodings = []  # holds multiple vectors, 1 for each image in the folder
            for img_name in folder_content:
                img_path = os.path.join(person_folder, img_name)
                if os.path.isfile(img_path) and (
                        img_name.endswith(".jpg") or img_name.endswith(".jpeg") or img_name.endswith(".png")):
                    # print(f"{img_name} - {img_path}")
                    encoding = img_path_to_encoding(img_path, FRmodel)
                    encodings.append(encoding)
            if len(encodings) > 0:  # only save record if we got at least 1 encoding
                if use_avg:  # use the average of all encodings (for faster runtimes when used in facial recognition)
                    avg = np.average(encodings,
                                     axis=0)  # find average for each value in the vector, maintains shape of original encoding
                    encodings = [avg]
                database[name] = encodings
                print(f"Facial Recognition: Entry for {name} added to database")
    return database


def load_database():
    """
    Read the database/database.pickle file in to memory
    :return: {"name": [tensor1....tensor_n]}
    """
    if os.path.exists(DATABASE_PATH):
        with open(DATABASE_PATH, "rb") as f:
            database = pickle.load(f)
        return database
    else:
        raise FileExistsError("Database file does not exist")


def who_is_it(image, database, threshold=0.52):
    """
    Make a prediction given an input image and a database with identities associated with their encodings
    :param image: image scanned for prediction
    :param database: dictionary with name to list of encoding pairs
    :param threshold: the maximum distance (0 to 1). Values less than this means a match was found
    :return: :return: {"identity": None/"a name", "distance": 0.0}
    """
    result = {"identity": None, "distance": 1}
    encoding = img_to_encoding(image, FRmodel)  # get encoding of scanned image (1x128 tensor)

    min_dist = 100
    identity = None

    # Loop over dictionary's names and encodings
    for name, encodings in database.items():
        for db_enc in encodings:
            dist = np.linalg.norm(db_enc - encoding)
            # print(f"Distance for {name} is {dist}")
            if dist < min_dist:
                min_dist = dist
                identity = name

    if min_dist < threshold:  # match found if less than set threshold
        result["identity"] = identity
        result["distance"] = min_dist
        # print(f"Distance for {name} is {dist}")
    return result


if __name__ == "__main__":
    pass
