"""
Code for dealing with the database.
"""

import os
import json
import random

import imageio
import numpy as np
from shutil import copy2


def data_type_block_dataset_from_structured_database(structured_database_directory, data_type_database_directory,
                                                     dataset_json_file_name):
    """
    Converts from the structured database to single file datasets per data type.

    :param structured_database_directory: The path to the structured database.
    :type structured_database_directory: str
    :param data_type_database_directory: The path where the single file per data type database should be placed.
    :type data_type_database_directory: str
    :param dataset_json_file_name: A JSON file containing the specifications of which parts of the structured database
                                   belong to which data type.
    :type dataset_json_file_name: str
    """
    with open(dataset_json_file_name) as json_file:
        dataset_dict = json.load(json_file)
    os.makedirs(data_type_database_directory, exist_ok=True)
    for data_type, cameras in dataset_dict.items():
        dataset_directory = os.path.join(data_type_database_directory, data_type)
        os.makedirs(dataset_directory, exist_ok=True)
        images = None
        labels = None
        rois = None
        perspectives = None
        for camera in cameras:
            camera_directory = os.path.join(structured_database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            if images is None:
                images = camera_images
                labels = camera_labels
                rois = np.tile(camera_roi, (labels.shape[0], 1, 1))
                perspectives = np.tile(camera_perspective, (labels.shape[0], 1, 1))
            else:
                images = np.concatenate((images, camera_images), axis=0)
                labels = np.concatenate((labels, camera_labels), axis=0)
                rois = np.concatenate((rois, np.tile(camera_roi, (camera_labels.shape[0], 1, 1))), axis=0)
                perspectives = np.concatenate((perspectives, np.tile(camera_perspective,
                                                                     (camera_labels.shape[0], 1, 1))), axis=0)
        np.save(os.path.join(dataset_directory, 'images.npy'), images)
        np.save(os.path.join(dataset_directory, 'labels.npy'), labels)
        np.save(os.path.join(dataset_directory, 'rois.npy'), rois)
        np.save(os.path.join(dataset_directory, 'perspectives.npy'), perspectives)


def generate_viable_camera_list(database_directory, dataset_json_file_name, viable_cameras_file_name):
    """
    Creates a JSON file containing a randomized order list of cameras which are from the training set and which have
    at least 20 labeled images.

    :param database_directory: The path to the database.
    :type database_directory: str
    :param dataset_json_file_name: A JSON file containing the specifications of which parts of the structured database
                                   belong to which data type.
    :type dataset_json_file_name: str
    :param viable_cameras_file_name: A JSON file to put the list of viable cameras in.
    :type viable_cameras_file_name: str
    """
    viable_camera_list = []
    with open(dataset_json_file_name) as json_file:
        dataset_dict = json.load(json_file)
    for camera_name in [directory for directory in os.listdir(database_directory) if not directory.startswith('.')]:
        if camera_name not in dataset_dict['train']:
            continue
        camera_directory = os.path.join(database_directory, camera_name)
        labels = np.load(os.path.join(camera_directory, 'labels.npy'), mmap_mode='r')
        example_count = labels.shape[0]
        if example_count < 20:
            continue
        viable_camera_list.append(camera_name)
    random.shuffle(viable_camera_list)
    with open(viable_cameras_file_name, 'w') as json_file:
        json.dump({'train': viable_camera_list}, json_file)


def specific_number_dataset_from_project_database(database_directory, dataset_directory,
                                                  dataset_json_file_name, number_of_cameras=1,
                                                  number_of_images_per_camera=1):
    """
    Converts from the structured database to single file datasets per data type.

    :param database_directory: The path to the structured database.
    :type database_directory: str
    :param dataset_directory: The path where the single file per data type database should be placed.
    :type dataset_directory: str
    :param dataset_json_file_name: A JSON file containing the specifications of which parts of the structured database
                                   belong to which data type.
    :type dataset_json_file_name: str
    :param number_of_cameras: The number of cameras to include.
    :type number_of_cameras: int
    :param number_of_images_per_camera: The number of images to use per camera.
    :type number_of_images_per_camera: int
    """
    with open(dataset_json_file_name) as json_file:
        dataset_dict = json.load(json_file)
    os.makedirs(dataset_directory, exist_ok=True)
    for data_type, cameras in dataset_dict.items():
        dataset_directory = os.path.join(dataset_directory, data_type)
        os.makedirs(dataset_directory, exist_ok=True)
        images = None
        labels = None
        rois = None
        perspectives = None
        unlabeled_video_writer = imageio.get_writer(os.path.join(dataset_directory, 'unlabeled_images.avi'), fps=50)
        for camera in cameras[:number_of_cameras]:
            camera_directory = os.path.join(database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            if images is None:
                images = camera_images[:number_of_images_per_camera]
                labels = camera_labels[:number_of_images_per_camera]
                rois = np.tile(camera_roi, (labels.shape[0], 1, 1))
                perspectives = np.tile(camera_perspective, (labels.shape[0], 1, 1))
            else:
                images = np.concatenate((images, camera_images[:number_of_images_per_camera]), axis=0)
                labels = np.concatenate((labels, camera_labels[:number_of_images_per_camera]), axis=0)
                rois = np.concatenate((rois, np.tile(camera_roi, (camera_labels.shape[0], 1, 1))), axis=0)
                perspectives = np.concatenate((perspectives, np.tile(camera_perspective,
                                                                     (camera_labels.shape[0], 1, 1))), axis=0)
            camera_unlabeled_directory = os.path.join(camera_directory, 'unlabeled')
            for file_name in os.listdir(camera_unlabeled_directory):
                if file_name.endswith('.avi'):
                    video_reader = imageio.get_reader(os.path.join(camera_unlabeled_directory, file_name))
                    for frame in video_reader:
                        unlabeled_video_writer.append_data(frame)
        unlabeled_video_writer.close()
        np.save(os.path.join(dataset_directory, 'images.npy'), images)
        np.save(os.path.join(dataset_directory, 'labels.npy'), labels)
        np.save(os.path.join(dataset_directory, 'rois.npy'), rois)
        np.save(os.path.join(dataset_directory, 'perspectives.npy'), perspectives)


data_type_block_dataset_from_structured_database('../storage/data/World Expo Database',
                                                 '../storage/data/World Expo Datasets',
                                                 '../storage/data/World Expo Database/datasets.json')

