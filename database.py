"""
Code to generate the database.
"""
import os
import numpy as np
import scipy.ndimage
import scipy.io
from PIL import Image, ImageDraw

head_standard_deviation_meters = 0.2


def original_database_to_project_database(original_directory, output_directory):
    train_label_directory = os.path.join(original_directory, 'train_label')
    train_frame_directory = os.path.join(original_directory, 'train_frame')
    train_perspective_directory = os.path.join(original_directory, 'train_perspective')
    os.makedirs(output_directory, exist_ok=True)
    for camera_name in os.listdir(train_label_directory):
        camera_directory = os.path.join(train_label_directory, camera_name)
        if os.path.isdir(camera_directory) and not camera_name.startswith('.'):
            perspective_path = os.path.join(train_perspective_directory, camera_name + '.mat')
            perspective = scipy.io.loadmat(perspective_path)['pMap'].astype(np.float32)
            roi_path = os.path.join(camera_directory, 'roi.mat')
            roi = generate_roi_array(roi_path, perspective)
            mat_list = [mat_file_name for mat_file_name in os.listdir(camera_directory)
                        if mat_file.endswith('.mat') and mat_file != 'roi.mat']
            images = None
            labels = None
            for index, mat_file in enumerate(mat_list):
                image_path = os.path.join(train_frame_directory, mat_file.replace('.mat', '.jpg'))
                head_positions_path = os.path.join(camera_directory, mat_file)
                head_positions = scipy.io.loadmat(head_positions_path)['point_position']
                label = generate_density_label(head_positions, perspective, roi)
                image = scipy.ndimage.imread(image_path)
                if not images:
                    images = np.zeros([len(mat_list), *image.shape], dtype=np.uint8)
                    labels = np.zeros([len(mat_list), image.shape[0], image.shape[1]], dtype=np.float32)
                images[index] = image
                labels[index] = label
            output_camera_directory = os.path.join(output_directory, camera_name)
            os.makedirs(output_camera_directory)
            np.save(os.path.join(output_camera_directory, 'images.npy'), images)
            np.save(os.path.join(output_camera_directory, 'labels.npy'), labels)
            np.save(os.path.join(output_camera_directory, 'perspective.npy'), perspective)
            np.save(os.path.join(output_camera_directory, 'roi.npy'), roi)


def generate_roi_array(roi_path, size_array):
    roi_mat = scipy.io.loadmat(roi_path)
    roi_x_list = roi_mat['maskVerticesXCoordinates'].flatten()
    roi_y_list = roi_mat['maskVerticesYCoordinates'].flatten()
    roi_vertex_list = zip(roi_y_list, roi_x_list)
    roi_image = Image.new('L', size_array.shape, 0)
    ImageDraw.Draw(roi_image).polygon(list(roi_vertex_list), outline=1, fill=1)
    roi = np.array(roi_image).astype(dtype=np.bool)
    return roi


def generate_density_label(head_positions, perspective, roi):
    """
    Generates a density label given the head positions and other meta data.

    :param head_positions: The head labeling positions.
    :type head_positions: np.ndarray
    :param perspective: The perspective map.
    :type perspective: np.ndarray
    :param roi: The region of interest map.
    :type roi: np.ndarray
    :return: The density labeling.
    :rtype: np.ndarray
    """
    label = np.zeros_like(perspective, dtype=np.float32)
    for head_position in head_positions:
        x, y = head_position
        position_perspective = perspective[y, x]
        head_standard_deviation = position_perspective * head_standard_deviation_meters
        head_gaussian = make_gaussian(head_standard_deviation)
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
                     ] = head_gaussian[y_start_offset:-y_end_offset, x_start_offset:-x_end_offset]
        person_label *= roi.astype(np.float32)
        normalized_person_label = person_label / person_label.sum()
        label += normalized_person_label
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
    except TypeError:
        x_off_center_size = int(standard_deviation * 2)
        y_off_center_size = int(standard_deviation * 2)
    x_linspace = np.linspace(-x_off_center_size, x_off_center_size, x_off_center_size * 2 + 1)
    y_linspace = np.linspace(-y_off_center_size, y_off_center_size, y_off_center_size * 2 + 1)
    x, y = np.meshgrid(x_linspace, y_linspace)
    d = np.sqrt(x * x + y * y)
    gaussian_array = np.exp(-(d**2 / 2.0))
    return gaussian_array


original_database_to_project_database(
    '/Users/golmschenk/Original World Expo Dataset',
    '/Users/golmschenk/World Expo Head Database'
)
