"""
Module with various logging utilities
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np


def get_pose_estimation_visualization(image: np.ndarray, joints: np.ndarray) -> np.ndarray:
    """
    Given an image and joints, return a new image that incldues visualizations
    of joints

    Args:
        image (np.ndarray): image on top of which to draw joints
        joints (np.ndarray): joints to draw on image

    Returns:
        np.ndarray: image with joints drawn on top
    """

    abandon_idxs = [0,1,14,15,16, 17]
    confidence_threshold = 0.1
    image = image.copy()
    circle_size = min(image.shape[:2]) // 100

    # draw joints
    for index, joint in enumerate(joints):

        if index in abandon_idxs:
            continue

        if joint[-1] > confidence_threshold:

            cv2.circle(
                image, (int(joint[0]), int(joint[1])),
                radius=circle_size, color=(255, 0, 0), thickness=-1)

    joint_thickness = min(image.shape[:2]) // 100

    part_orders = [
        (2, 5), (5, 11), (2, 8), (8, 11), (5, 6), (6, 7), (2, 3), (3, 4), (11, 12), (12, 13), (8, 9),
        (9, 10)
    ]

    color_map = plt.get_cmap('rainbow')
    colors = [255 * np.array(color) for color in color_map(np.linspace(0, 1, len(part_orders)))]

    # draw link
    for index, pair in enumerate(part_orders):

        if joints[pair[0]][-1] > confidence_threshold and joints[pair[1]][-1] > confidence_threshold:
            cv2.line( image,
                (int(joints[pair[0]][0]), int(joints[pair[0]][1])),
                (int(joints[pair[1]][0]), int(joints[pair[1]][1])),
                colors[index], joint_thickness)

    return image
