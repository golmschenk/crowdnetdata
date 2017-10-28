"""
Code for preparing the unlabeled data from the original dataset.
"""

import os
from shutil import copy2


def move_videos_to_project_database(original_database_directory, project_database_directory):
    """
    Moves the videos from the original database into the project database format.

    :param original_database_directory: The path to the original database directory.
    :type original_database_directory: str
    :param project_database_directory: The path to the project database directory.
    :type project_database_directory: str
    """
    video_directories = [os.path.join(original_database_directory, 'train_video_part1'),
                         os.path.join(original_database_directory, 'train_video_part2')]
    for video_directory in video_directories:
        video_list = [name for name in os.listdir(video_directory) if name.endswith('.avi')]
        video_prefixes = list(set([video_name.split('_')[0] for video_name in video_list]))
        for prefix in video_prefixes:
            camera_video_list = [video_name for video_name in video_list if video_name.startswith(prefix)]
            for video_index, video_name in enumerate(camera_video_list):
                unlabeled_directory = os.path.join(project_database_directory, prefix, 'unlabeled')
                os.makedirs(unlabeled_directory, exist_ok=True)
                copy2(os.path.join(video_directory, video_name),
                      os.path.join(unlabeled_directory, '{}.avi'.format(video_index)))
    test_video_directory = os.path.join(original_database_directory, 'test_video')
    video_list = [name for name in os.listdir(test_video_directory) if name.endswith('.avi')]
    video_prefixes = list(set([video_name.split('_')[0] for video_name in video_list]))
    for prefix in video_prefixes:
        camera_video_list = [video_name for video_name in video_list if video_name.startswith(prefix)]
        for video_index, video_name in enumerate(camera_video_list):
            unlabeled_directory = os.path.join(project_database_directory, 'test_{}'.format(prefix), 'unlabeled')
            os.makedirs(unlabeled_directory, exist_ok=True)
            copy2(os.path.join(test_video_directory, video_name),
                  os.path.join(unlabeled_directory, '{}.avi'.format(video_index)))
