"""
Code for inspecting the data.
"""

import numpy as np
import csv
import os


def generate_csv_of_camera_statistics(database_directory, csv_output_path):
    """
    Script to get the camera statistics in a CSV file for examining.

    :param database_directory: The directory path of the database.
    :type database_directory: str
    :param csv_output_path: The path to put the statistics file at.
    :type csv_output_path: str
    """
    with open(csv_output_path, 'w') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Camera', 'Example Count', 'Mean Person Count', 'Unlabeled Video Count'])
        for camera_name in [directory for directory in os.listdir(database_directory) if not directory.startswith('.')]:
            camera_directory = os.path.join(database_directory, camera_name)
            labels = np.load(os.path.join(camera_directory, 'labels.npy'), mmap_mode='r')
            example_count = labels.shape[0]
            mean_person_count = labels.sum(axis=(1,2)).mean()
            video_count = len([video for video in os.listdir(os.path.join(camera_directory, 'unlabeled'))
                               if video.endswith('.avi')])
            writer.writerow([camera_name, example_count, mean_person_count, video_count])
