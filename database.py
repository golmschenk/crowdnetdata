"""
Code to generate the database.
"""
import os
import h5py
import numpy as np
import scipy.ndimage
import scipy.io
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


def generate_density_label(head_positions, perspective):
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
        head_count += 1
        x, y = head_position.astype(np.uint32)
        if 0 <= x < perspective.shape[1]:
            position_perspective = perspective[y, x]
        else:
            position_perspective = perspective[y, 0]
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
                     ] = head_gaussian[y_start_offset:head_gaussian.shape[0] - y_end_offset,
                                       x_start_offset:head_gaussian.shape[1] - x_end_offset]
        body_x = x
        body_y = y + (position_perspective * body_height_offset_meters)
        body_width_standard_deviation = position_perspective * body_width_standard_deviation_meters
        body_height_standard_deviation = position_perspective * body_height_standard_deviation_meters
        body_gaussian = make_gaussian((body_width_standard_deviation, body_height_standard_deviation))
        body_gaussian = body_gaussian / (2 * body_gaussian.sum())
        person_label = np.zeros_like(label, dtype=np.float32)
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
                     ] = body_gaussian[y_start_offset:body_gaussian.shape[0] - y_end_offset,
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
        x_off_center_size = int(standard_deviation[0] * 2)
        y_off_center_size = int(standard_deviation[1] * 2)
    except (IndexError, TypeError):
        x_off_center_size = int(standard_deviation * 2)
        y_off_center_size = int(standard_deviation * 2)
    x_linspace = np.linspace(-x_off_center_size, x_off_center_size, x_off_center_size * 2 + 1)
    y_linspace = np.linspace(-y_off_center_size, y_off_center_size, y_off_center_size * 2 + 1)
    x, y = np.meshgrid(x_linspace, y_linspace)
    d = np.sqrt(x * x + y * y)
    gaussian_array = np.exp(-(d**2 / (2.0 * standard_deviation ** 2)))
    return gaussian_array


original_database_to_project_database(
    '/Users/golmschenk/Original World Expo Dataset',
    '/Users/golmschenk/Head World Expo Database'
)

print('{} head positions identified.'.format(head_count))
