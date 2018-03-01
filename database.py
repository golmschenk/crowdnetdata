"""
Code to generate the database.
"""
import os
import re
import math
import h5py
import numpy as np
import pandas.io.parsers
import scipy.ndimage
import scipy.io
import sys
from PIL import Image, ImageDraw

from incorrectly_labeled import incorrectly_labeled

head_standard_deviation_meters = 0.2
body_width_standard_deviation_meters = 0.2
body_height_standard_deviation_meters = 0.5
body_height_offset_meters = 0.875
head_count = 0


def original_database_to_project_database(original_directory, output_directory):
    """
    Creates the project format database from the original World Expo database.

    :param original_directory: The directory containing the original World Expo database.
    :type original_directory: str
    :param output_directory: The directory to save the new database.
    :type output_directory: str
    """
    for data_type in ['train', 'test']:
        label_directory = os.path.join(original_directory, '{}_label'.format(data_type))
        frame_directory = os.path.join(original_directory, '{}_frame'.format(data_type))
        perspective_directory = os.path.join(original_directory, '{}_perspective'.format(data_type))
        os.makedirs(output_directory, exist_ok=True)
        for camera_name in os.listdir(label_directory):
            camera_directory = os.path.join(label_directory, camera_name)
            if os.path.isdir(camera_directory) and not camera_name.startswith('.'):
                print('Processing camera {}...'.format(camera_name))
                perspective_path = os.path.join(perspective_directory, camera_name + '.mat')
                perspective = load_mat(perspective_path)['pMap'].astype(np.float32)
                roi_path = os.path.join(camera_directory, 'roi.mat')
                roi = generate_roi_array(roi_path, perspective)
                mat_list = [mat_file_name for mat_file_name in os.listdir(camera_directory)
                            if mat_file_name.endswith('.mat') and mat_file_name != 'roi.mat' and
                            mat_file_name.replace('.mat', '') not in incorrectly_labeled]
                images = None
                labels = None
                for index, mat_file in enumerate(mat_list):
                    if data_type == 'test':
                        image_path = os.path.join(frame_directory, camera_name, mat_file.replace('.mat', '.jpg'))
                    else:
                        image_path = os.path.join(frame_directory, mat_file.replace('.mat', '.jpg'))
                    head_positions_path = os.path.join(camera_directory, mat_file)
                    head_positions = load_mat(head_positions_path)['point_position']
                    label = generate_density_label(head_positions, perspective)
                    image = scipy.ndimage.imread(image_path)
                    if images is None:
                        images = np.zeros([len(mat_list), *image.shape], dtype=np.uint8)
                        labels = np.zeros([len(mat_list), image.shape[0], image.shape[1]], dtype=np.float32)
                    images[index] = image
                    labels[index] = label
                output_prefix = 'test_' if data_type == 'test' else ''
                output_camera_directory = os.path.join(output_directory, output_prefix + camera_name)
                os.makedirs(output_camera_directory, exist_ok=True)
                if data_type == 'test' and images.shape[0] == 119:
                    images = np.append(images, np.expand_dims(images[-1], axis=0), axis=0)
                    labels = np.append(labels, np.expand_dims(labels[-1], axis=0), axis=0)
                np.save(os.path.join(output_camera_directory, 'images.npy'), images)
                np.save(os.path.join(output_camera_directory, 'labels.npy'), labels)
                np.save(os.path.join(output_camera_directory, 'perspective.npy'), perspective)
                np.save(os.path.join(output_camera_directory, 'roi.npy'), roi)


def hao_ben_simulated_data_to_project_database(original_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)
    image_directory = os.path.join(original_directory, 'images')
    perspective_directory = os.path.join(original_directory, 'perspective map')
    bounding_box_directory = os.path.join(original_directory, 'bounding_boxes')
    image_list = []
    perspective_list = []
    roi_list = []
    label_list = []
    print('{} mat available.'.format(len(os.listdir(perspective_directory))))
    mat_used_count = 0
    for perspective_file_name in os.listdir(perspective_directory):
        if perspective_file_name.endswith('.mat'):
            file_number = int(perspective_file_name.replace('pmap', '').replace('.mat', ''))
            perspective = load_mat(os.path.join(perspective_directory, perspective_file_name))['img_data'].astype(np.float32)
            perspective /= 1.75
            if perspective.max() > 500 or perspective[perspective.shape[0]//2, 0] < 0 or perspective[-1, 0] < 0:
                continue
            perspective = np.maximum(perspective, 3)
            roi = np.ones_like(perspective, dtype=np.bool)
            image = scipy.ndimage.imread(os.path.join(image_directory, '{}.png'.format(file_number)))
            image = image[:, :, :3]
            bounding_boxes = np.genfromtxt(os.path.join(bounding_box_directory, 'bb{}.csv'.format(file_number)), delimiter=',')
            head_positions = np.stack((((bounding_boxes[:, 1] + bounding_boxes[:, 3]) / 2), bounding_boxes[:, 2]), axis=1).astype(np.int32)
            label = generate_density_label(head_positions, perspective, ignore_tiny=True)
            image_list.append(image)
            label_list.append(label)
            perspective_list.append(perspective)
            roi_list.append(roi)
            mat_used_count += 1
    print('{} mat used.'.format(mat_used_count))
    images = np.stack(image_list)
    labels = np.stack(label_list)
    rois = np.stack(roi_list)
    perspectives = np.stack(perspective_list)
    np.save(os.path.join(output_directory, 'images.npy'), images)
    np.save(os.path.join(output_directory, 'labels.npy'), labels)
    np.save(os.path.join(output_directory, 'perspectives.npy'), perspectives)
    np.save(os.path.join(output_directory, 'rois.npy'), rois)


def split_existing_data(data_directory):
    os.makedirs(os.path.join(data_directory, 'train'), exist_ok=True)
    os.makedirs(os.path.join(data_directory, 'validation'), exist_ok=True)
    images = np.load(os.path.join(data_directory, 'images.npy'))
    images0 = images[:images.shape[0] // 2]
    images1 = images[images.shape[0] // 2:]
    np.save(os.path.join(data_directory, 'train', 'images.npy'), images0)
    np.save(os.path.join(data_directory, 'validation', 'images.npy'), images1)
    labels = np.load(os.path.join(data_directory, 'labels.npy'))
    labels0 = labels[:labels.shape[0] // 2]
    labels1 = labels[labels.shape[0] // 2:]
    np.save(os.path.join(data_directory, 'train', 'labels.npy'), labels0)
    np.save(os.path.join(data_directory, 'validation', 'labels.npy'), labels1)
    rois = np.load(os.path.join(data_directory, 'rois.npy'))
    rois0 = rois[:rois.shape[0] // 2]
    rois1 = rois[rois.shape[0] // 2:]
    np.save(os.path.join(data_directory, 'train', 'rois.npy'), rois0)
    np.save(os.path.join(data_directory, 'validation', 'rois.npy'), rois1)
    perspectives = np.load(os.path.join(data_directory, 'perspectives.npy'))
    perspectives0 = perspectives[:perspectives.shape[0] // 2]
    perspectives1 = perspectives[perspectives.shape[0] // 2:]
    np.save(os.path.join(data_directory, 'train', 'perspectives.npy'), perspectives0)
    np.save(os.path.join(data_directory, 'validation', 'perspectives.npy'), perspectives1)


def read_hao_ben_meta_file(file_path):
    image_head_list = []
    head_positions_dict = {}
    height_dict = {}

    with open(file_path) as file:
        pictures_regex = re.compile(r'\w+/(\w+\.png)\s+(\d+)')
        persons_regex = re.compile(r'(\d+)\s+(\d+\.\d+)\s+\[(\d+),\s*(\d+)]')

        for line in file:
            if line.isspace():
                continue
            line = line.strip()
            match = pictures_regex.match(line)
            if match:
                file, id = match.group(1), int(match.group(2))
                image_head_list.append((file, id))
                continue
            match = persons_regex.match(line)
            if match:
                person, x, y = map(int, match.group(1, 3, 4))
                height = float(match.group(2))
                head_positions_dict[person] = (x, y)
                height_dict[person] = height
                continue
    return image_head_list, head_positions_dict, height_dict


def load_mat(mat_file_path):
    """
    Load either format of mat files the same way.

    :param mat_file_path: The path to the mat file.
    :type mat_file_path: str
    :return: The loaded mat dict.
    :rtype: dict[T]
    """
    try:
        return scipy.io.loadmat(mat_file_path)
    except NotImplementedError:
        with h5py.File(mat_file_path, 'r') as f:
            return {key: value.value.transpose() for (key, value) in f.items()}


def generate_roi_array(roi_path, size_array):
    """
    Generates the ROI array based on the polygon information.

    :param roi_path: The path to the polygon information of the ROI.
    :type roi_path: str
    :param size_array: An array to base the size of the ROI array on.
    :type size_array: np.ndarray
    :return: The ROI array.
    :rtype: np.ndarray
    """
    roi_mat = load_mat(roi_path)
    roi_x_list = roi_mat['maskVerticesXCoordinates'].flatten()
    roi_y_list = roi_mat['maskVerticesYCoordinates'].flatten()
    roi_vertex_list = zip(roi_y_list, roi_x_list)
    roi_image = Image.new('L', size_array.shape, 0)
    ImageDraw.Draw(roi_image).polygon(list(roi_vertex_list), outline=1, fill=1)
    roi = np.array(roi_image).astype(dtype=np.bool).transpose()
    return roi


def generate_density_label(head_positions, perspective, ignore_tiny=False):
    """
    Generates a density label given the head positions and other meta data.

    :param head_positions: The head labeling positions.
    :type head_positions: np.ndarray
    :param perspective: The perspective map.
    :type perspective: np.ndarray
    :return: The density labeling.
    :rtype: np.ndarray
    """
    global head_count
    label = np.zeros_like(perspective, dtype=np.float32)
    for head_position in head_positions:
        x, y = head_position.astype(np.uint32)
        if 0 <= x < perspective.shape[1]:
            position_perspective = perspective[y, x]
        else:
            position_perspective = perspective[y, 0]
        if ignore_tiny and position_perspective < 3.1:
            continue
        head_count += 1
        head_standard_deviation = position_perspective * head_standard_deviation_meters
        head_gaussian = make_gaussian(head_standard_deviation)
        head_gaussian = head_gaussian / (2 * head_gaussian.sum())
        person_label = np.zeros_like(label, dtype=np.float32)
        off_center_size = int((head_gaussian.shape[0] - 1) / 2)
        y_start_offset = 0
        if y - off_center_size < 0:
            y_start_offset = off_center_size - y
        y_end_offset = 0
        if y + off_center_size >= person_label.shape[0]:
            y_end_offset = (y + off_center_size + 1) - person_label.shape[0]
        x_start_offset = 0
        if x - off_center_size < 0:
            x_start_offset = off_center_size - x
        x_end_offset = 0
        if x + off_center_size >= person_label.shape[1]:
            x_end_offset = (x + off_center_size + 1) - person_label.shape[1]
        person_label[y - off_center_size + y_start_offset:y + off_center_size + 1 - y_end_offset,
                     x - off_center_size + x_start_offset:x + off_center_size + 1 - x_end_offset
                     ] += head_gaussian[y_start_offset:head_gaussian.shape[0] - y_end_offset,
                                        x_start_offset:head_gaussian.shape[1] - x_end_offset]
        body_x = x
        body_y = y + int(position_perspective * body_height_offset_meters)
        body_width_standard_deviation = position_perspective * body_width_standard_deviation_meters
        body_height_standard_deviation = position_perspective * body_height_standard_deviation_meters
        body_gaussian = make_gaussian((body_width_standard_deviation, body_height_standard_deviation))
        body_gaussian = body_gaussian / (2 * body_gaussian.sum())
        x_off_center_size = int((body_gaussian.shape[1] - 1) / 2)
        y_off_center_size = int((body_gaussian.shape[0] - 1) / 2)
        y_start_offset = 0
        if body_y - y_off_center_size < 0:
            y_start_offset = y_off_center_size - body_y
        y_end_offset = 0
        if body_y + y_off_center_size >= person_label.shape[0]:
            y_end_offset = (body_y + y_off_center_size + 1) - person_label.shape[0]
        x_start_offset = 0
        if body_x - x_off_center_size < 0:
            x_start_offset = x_off_center_size - body_x
        x_end_offset = 0
        if body_x + x_off_center_size >= person_label.shape[1]:
            x_end_offset = (body_x + x_off_center_size + 1) - person_label.shape[1]
        person_label[body_y - y_off_center_size + y_start_offset:body_y + y_off_center_size + 1 - y_end_offset,
                     body_x - x_off_center_size + x_start_offset:body_x + x_off_center_size + 1 - x_end_offset
                     ] += body_gaussian[y_start_offset:body_gaussian.shape[0] - y_end_offset,
                                        x_start_offset:body_gaussian.shape[1] - x_end_offset]
        label += person_label
    return label


def make_gaussian(standard_deviation=1.0):
    """
    Make a square gaussian kernel.

    :param standard_deviation: The standard deviation of the 2D gaussian.
    :type standard_deviation: float | (float, float)
    :return: The gaussian array.
    :rtype: np.ndarray
    """
    try:
        x_standard_deviation = standard_deviation[0]
        y_standard_deviation = standard_deviation[1]
    except (IndexError, TypeError):
        x_standard_deviation = standard_deviation
        y_standard_deviation = standard_deviation
    x_off_center_size = int(x_standard_deviation * 2)
    y_off_center_size = int(y_standard_deviation * 2)
    x_linspace = np.linspace(-x_off_center_size, x_off_center_size, x_off_center_size * 2 + 1)
    y_linspace = np.linspace(-y_off_center_size, y_off_center_size, y_off_center_size * 2 + 1)
    x, y = np.meshgrid(x_linspace, y_linspace)
    x_part = (x ** 2) / (2.0 * x_standard_deviation ** 2)
    y_part = (y ** 2) / (2.0 * y_standard_deviation ** 2)
    gaussian_array = np.exp(-(x_part + y_part))
    return gaussian_array


original_database_to_project_database(
    '/mnt/Gold/data/World Expo Original',
    '/mnt/Gold/data/World Expo'
)

# hao_ben_simulated_data_to_project_database('/Volumes/Gold/Datasets/LCrowdV/Hao Ben', '/Users/golmschenk/Downloads')
# split_existing_data('/Users/golmschenk/Downloads')

