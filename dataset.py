"""
Code for dealing with the database.
"""

import os
import json
import random
import math

import imageio
import numpy as np
import shutil


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


def dummy_dataset_from_video(video_directory, output_directory, start_frame=0, end_frame=math.inf, every_nth_frame=1):
    video_reader = imageio.get_reader(os.path.join(video_directory, 'video.avi'))
    print('FPS: {}'.format(video_reader.get_meta_data()['fps']))
    image_list = []
    for index, image in enumerate(video_reader):
        if index >= start_frame and index < end_frame and index % every_nth_frame == 0:
            image_list.append(image)
    images = np.stack(image_list)
    labels = np.zeros(shape=images.shape[:3], dtype=np.float32)
    roi = np.load(os.path.join(video_directory, 'roi.npy'))
    rois = np.tile(roi, (labels.shape[0], 1, 1))
    perspective = np.load(os.path.join(video_directory, 'perspective.npy'))
    perspectives = np.tile(perspective, (labels.shape[0], 1, 1))
    os.makedirs(output_directory, exist_ok=True)
    np.save(os.path.join(output_directory, 'images.npy'), images)
    np.save(os.path.join(output_directory, 'labels.npy'), labels)
    np.save(os.path.join(output_directory, 'rois.npy'), rois)
    np.save(os.path.join(output_directory, 'perspectives.npy'), perspectives)


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


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def specific_number_dataset_from_project_database(database_directory, dataset_directory,
                                                  dataset_json_file_name, number_of_cameras=1,
                                                  number_of_images_per_camera=1):
    """
    Converts from the structured database to single file datasets per data type with a specific number of
    images and cameras, along with the related unlabeled data.

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
        unlabeled_estimates = None
        unlabeled_image_counts = None
        unlabeled_perspectives = None
        unlabeled_rois = None
        unlabeled_video_writer = imageio.get_writer(os.path.join(dataset_directory, 'unlabeled_images.avi'), fps=50)
        for camera in cameras[:number_of_cameras]:
            camera_directory = os.path.join(database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            camera_unlabeled_directory = os.path.join(camera_directory, 'unlabeled')
            camera_unlabeled_image_count = 0
            for file_name in os.listdir(camera_unlabeled_directory):
                if file_name.endswith('.avi'):
                    video_reader = imageio.get_reader(os.path.join(camera_unlabeled_directory, file_name))
                    for frame in video_reader:
                        unlabeled_video_writer.append_data(frame)
                        camera_unlabeled_image_count += 1
            if images is None:
                images = camera_images[:number_of_images_per_camera]
                labels = camera_labels[:number_of_images_per_camera]
                rois = np.tile(camera_roi, (labels.shape[0], 1, 1))
                perspectives = np.tile(camera_perspective, (labels.shape[0], 1, 1))
                unlabeled_estimates = np.array([camera_labels[:number_of_images_per_camera].sum(axis=(1, 2)).mean()],
                                               dtype=np.float32)
                unlabeled_image_counts = np.array([camera_unlabeled_image_count], dtype=np.int32)
                unlabeled_perspectives = np.expand_dims(camera_perspective, axis=0)
                unlabeled_rois = np.expand_dims(camera_roi, axis=0)
            else:
                images = np.concatenate((images, camera_images[:number_of_images_per_camera]), axis=0)
                labels = np.concatenate((labels, camera_labels[:number_of_images_per_camera]), axis=0)
                rois = np.concatenate((rois, np.tile(camera_roi, (camera_labels.shape[0], 1, 1))), axis=0)
                perspectives = np.concatenate((perspectives, np.tile(camera_perspective,
                                                                     (camera_labels.shape[0], 1, 1))), axis=0)
                camera_unlabeled_estimate = camera_labels[:number_of_images_per_camera].sum(axis=(1, 2)).mean()
                unlabeled_estimates = np.append(unlabeled_estimates, camera_unlabeled_estimate)
                unlabeled_image_counts = np.append(unlabeled_image_counts, camera_unlabeled_image_count)
                unlabeled_perspectives = np.append(unlabeled_perspectives, [camera_perspective], axis=0)
                unlabeled_rois = np.append(unlabeled_rois, [camera_roi], axis=0)
        unlabeled_video_writer.close()
        np.save(os.path.join(dataset_directory, 'images.npy'), images)
        np.save(os.path.join(dataset_directory, 'labels.npy'), labels)
        np.save(os.path.join(dataset_directory, 'rois.npy'), rois)
        np.save(os.path.join(dataset_directory, 'perspectives.npy'), perspectives)
        np.save(os.path.join(dataset_directory, 'unlabeled_image_counts.npy'), unlabeled_image_counts)
        np.save(os.path.join(dataset_directory, 'unlabeled_rois.npy'), unlabeled_rois)
        np.save(os.path.join(dataset_directory, 'unlabeled_perspectives.npy'), unlabeled_perspectives)


def generate_systematic_datasets(database_directory, dataset_root_directory, dataset_json_file_name):
    """
    Generates a specified set of datasets

    :param database_directory: The path to the structured database.
    :type database_directory: str
    :param dataset_root_directory: The path to put the datasets.
    :type dataset_root_directory: str
    :param dataset_json_file_name: A JSON file containing the specifications of which parts of the structured database
                                   belong to which data type.
    :type dataset_json_file_name: str
    """
    os.makedirs(dataset_root_directory, exist_ok=True)
    camera_count_list = [1, 3, 5, 10, 20]
    image_count_list = [1, 3, 5, 10, 20]
    for camera_count in camera_count_list:
        for image_count in image_count_list:
            dataset_name = '{} Cameras {} Images'.format(camera_count, image_count)
            dataset_directory = os.path.join(dataset_root_directory, dataset_name)
            specific_number_dataset_from_project_database(database_directory, dataset_directory, dataset_json_file_name,
                                                          camera_count, image_count)


def specific_number_dataset_from_project_database_using_target_unlabeled(database_directory, dataset_directory,
                                                                         dataset_json_file_name, number_of_cameras=1,
                                                                         number_of_images_per_camera=1,
                                                                         remove_test_and_validation=False):
    """
    Converts from the structured database to single file datasets per data type with a specific number of
    images and cameras, along with the related unlabeled data.

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
    train_directory = os.path.join(dataset_directory, 'train')
    os.makedirs(train_directory, exist_ok=True)
    unlabeled_perspectives = None
    unlabeled_rois = None
    unlabeled_image_counts = None
    unlabeled_video_writer = imageio.get_writer(os.path.join(train_directory, 'unlabeled_images.avi'), fps=50)
    for data_type, cameras in dataset_dict.items():
        if data_type == 'train':
            cameras = cameras[:number_of_cameras]
        data_type_unlabeled_image_count = 0
        data_type_directory = os.path.join(dataset_directory, data_type)
        os.makedirs(data_type_directory, exist_ok=True)
        images = None
        labels = None
        rois = None
        perspectives = None
        for camera in cameras:
            print('Processing camera {}...'.format(camera))
            camera_directory = os.path.join(database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            if data_type == 'train':
                cameras = cameras[:number_of_cameras]
                camera_images = camera_images[:number_of_images_per_camera]
                camera_labels = camera_labels[:number_of_images_per_camera]
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            camera_unlabeled_image_count = 0
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
            if data_type == 'validation' or data_type == 'test':
                camera_unlabeled_directory = os.path.join(camera_directory, 'unlabeled')
                for file_name in os.listdir(camera_unlabeled_directory):
                    if file_name.endswith('.avi'):
                        video_reader = imageio.get_reader(os.path.join(camera_unlabeled_directory, file_name))
                        for frame_index, frame in enumerate(video_reader):
                            if (data_type == 'test' and frame_index % 500 == 0) or (
                                    data_type == 'validation' and frame_index % 50 == 0):
                                unlabeled_video_writer.append_data(frame)
                                data_type_unlabeled_image_count += 1
                                camera_unlabeled_image_count += 1
                if unlabeled_perspectives is None:
                    unlabeled_perspectives = np.expand_dims(camera_perspective, axis=0)
                    unlabeled_rois = np.expand_dims(camera_roi, axis=0)
                    unlabeled_image_counts = np.array([camera_unlabeled_image_count], dtype=np.int32)
                else:
                    unlabeled_perspectives = np.append(unlabeled_perspectives, [camera_perspective], axis=0)
                    unlabeled_rois = np.append(unlabeled_rois, [camera_roi], axis=0)
                    unlabeled_image_counts = np.append(unlabeled_image_counts, camera_unlabeled_image_count)
        np.save(os.path.join(data_type_directory, 'images.npy'), images)
        np.save(os.path.join(data_type_directory, 'labels.npy'), labels)
        np.save(os.path.join(data_type_directory, 'rois.npy'), rois)
        np.save(os.path.join(data_type_directory, 'perspectives.npy'), perspectives)
        print('`{}` added {} unlabeled images.'.format(data_type, data_type_unlabeled_image_count))
    unlabeled_video_writer.close()
    np.save(os.path.join(train_directory, 'unlabeled_rois.npy'), unlabeled_rois)
    np.save(os.path.join(train_directory, 'unlabeled_perspectives.npy'), unlabeled_perspectives)
    np.save(os.path.join(train_directory, 'unlabeled_image_counts.npy'), unlabeled_image_counts)
    if remove_test_and_validation:
        shutil.rmtree(os.path.join(dataset_directory, 'test'))
        shutil.rmtree(os.path.join(dataset_directory, 'validation'))


def specific_number_dataset_from_project_database_using_target_unlabeled_numpy(database_directory, dataset_directory,
                                                                         dataset_json_file_name, number_of_cameras=1,
                                                                         number_of_images_per_camera=1,
                                                                         remove_test_and_validation=False):
    """
    Converts from the structured database to single file datasets per data type with a specific number of
    images and cameras, along with the related unlabeled data.

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
    train_directory = os.path.join(dataset_directory, 'train')
    os.makedirs(train_directory, exist_ok=True)
    unlabeled_perspectives = None
    unlabeled_rois = None
    unlabeled_images = None
    for data_type, cameras in dataset_dict.items():
        if data_type == 'train':
            cameras = cameras[:number_of_cameras]
        data_type_unlabeled_image_count = 0
        data_type_directory = os.path.join(dataset_directory, data_type)
        os.makedirs(data_type_directory, exist_ok=True)
        images = None
        labels = None
        rois = None
        perspectives = None
        for camera in cameras:
            print('Processing camera {}...'.format(camera))
            camera_directory = os.path.join(database_directory, camera)
            camera_images = np.load(os.path.join(camera_directory, 'images.npy'))
            camera_labels = np.load(os.path.join(camera_directory, 'labels.npy'))
            if data_type == 'train':
                cameras = cameras[:number_of_cameras]
                camera_images = camera_images[:number_of_images_per_camera]
                camera_labels = camera_labels[:number_of_images_per_camera]
            camera_roi = np.load(os.path.join(camera_directory, 'roi.npy'))
            camera_perspective = np.load(os.path.join(camera_directory, 'perspective.npy'))
            camera_unlabeled_image_count = 0
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
            if data_type == 'validation' or data_type == 'test':
                camera_unlabeled_directory = os.path.join(camera_directory, 'unlabeled')
                for file_name in os.listdir(camera_unlabeled_directory):
                    if file_name.endswith('.avi'):
                        video_reader = imageio.get_reader(os.path.join(camera_unlabeled_directory, file_name))
                        for frame_index, frame in enumerate(video_reader):
                            if (data_type == 'test' and frame_index % 500 == 0) or (
                                    data_type == 'validation' and frame_index % 50 == 0):
                                if unlabeled_images is None:
                                    unlabeled_images = np.expand_dims(frame, axis=0)
                                    unlabeled_perspectives = np.expand_dims(camera_perspective, axis=0)
                                    unlabeled_rois = np.expand_dims(camera_roi, axis=0)
                                else:
                                    unlabeled_images = np.append(unlabeled_images, [frame], axis=0)
                                    unlabeled_perspectives = np.append(unlabeled_perspectives, [camera_perspective], axis=0)
                                    unlabeled_rois = np.append(unlabeled_rois, [camera_roi], axis=0)
                                data_type_unlabeled_image_count += 1
                                camera_unlabeled_image_count += 1
        np.save(os.path.join(data_type_directory, 'images.npy'), images)
        np.save(os.path.join(data_type_directory, 'labels.npy'), labels)
        np.save(os.path.join(data_type_directory, 'rois.npy'), rois)
        np.save(os.path.join(data_type_directory, 'perspectives.npy'), perspectives)
        print('`{}` added {} unlabeled images.'.format(data_type, data_type_unlabeled_image_count))
    np.save(os.path.join(train_directory, 'unlabeled_images.npy'), unlabeled_images)
    np.save(os.path.join(train_directory, 'unlabeled_rois.npy'), unlabeled_rois)
    np.save(os.path.join(train_directory, 'unlabeled_perspectives.npy'), unlabeled_perspectives)
    if remove_test_and_validation:
        shutil.rmtree(os.path.join(dataset_directory, 'test'))
        shutil.rmtree(os.path.join(dataset_directory, 'validation'))


# dummy_dataset_from_video('/Users/golmschenk/Desktop/test_200608',
#                         '/Users/golmschenk/Desktop/200608 Time Lapse Demo',
#                          every_nth_frame=60)

# specific_number_dataset_from_project_database('/Volumes/Gold/crowd/datasets/World Expo/World Expo Database',
#                              '/Volumes/Gold/crowd/datasets/World Expo/Unlabeled World Expo Datasets/All Cameras All Images',
#                              '/Volumes/Gold/crowd/datasets/World Expo/train_only.json', None, None)

for camera_count in [5, 1, 3, 10, 20]:
    for image_count in [5, 1, 3, 10, 20]:
        print('Processing {} Camera {} Images...'.format(camera_count, image_count))
        specific_number_dataset_from_project_database_using_target_unlabeled_numpy(
            '/media/root/Gold/crowd/data/World Expo', '/media/root/Gold/crowd/data/World Expo Datasets/{} Camera {} Images Target Unlabeled'.format(camera_count, image_count),
            '/media/root/Gold/crowd/data/World Expo/viable_with_validation_and_random_test.json', number_of_cameras=camera_count, number_of_images_per_camera=image_count, remove_test_and_validation=True
        )

# specific_number_dataset_from_project_database_using_target_unlabeled_numpy(
#     '/media/root/Gold/crowd/data/World Expo', '/media/root/Gold/crowd/data/World Expo Datasets/Test And Validation',
#     '/media/root/Gold/crowd/data/World Expo/viable_with_validation_and_random_test.json', number_of_cameras=1, number_of_images_per_camera=1, remove_test_and_validation=False
# )