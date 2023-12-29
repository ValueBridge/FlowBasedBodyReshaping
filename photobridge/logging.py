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


def get_flow_visualization(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Visualize flow on image. Use jet colormap for flow.

    Args:
        image (np.ndarray): image to visualize flow on
        flow (np.ndarray): flow

    Returns:
        np.ndarray: image with flow visualized
    """

    image = image.astype(np.float32)

    absolute_flow = np.abs(flow)

    flow_colormap = cv2.applyColorMap(absolute_flow.astype(np.uint8), cv2.COLORMAP_JET).astype(np.float32)

    flow_weight = (absolute_flow / np.max(absolute_flow)).astype(np.float32)

    # Repeat flow weight to three channels
    flow_weight = np.repeat(flow_weight[:, :, np.newaxis], 3, axis=2)

    # Overlay weighted flow colormap over original image
    overlay = (image * (1 - flow_weight)) + (flow_colormap * flow_weight)

    return np.clip(overlay, 0, 255)


def get_flow_visualization_combined(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """
    Visualize combined x and y flows on image.

    Args:
        image (np.ndarray): image to visualize flow on
        flow (np.ndarray): m x n x 2 flow tensor. First channel contains x flow, second channel contains y flow

    Returns:
        np.ndarray: image with flow visualized
    """

    image = image.astype(np.float32)

    absolute_flow = np.abs(flow)

    x_flow = absolute_flow[:, :, 0]
    y_flow = absolute_flow[:, :, 1]

    combined_flow = x_flow + y_flow
    scaled_combined_flow = combined_flow / np.max(combined_flow)

    epsilon = 1e-6

    scaled_x_flow = x_flow / (np.max(x_flow) + epsilon)
    scaled_y_flow = y_flow / (np.max(y_flow) + epsilon)

    x_flow_colormap = cv2.applyColorMap(
        (255 * scaled_x_flow).astype(np.uint8),
        cv2.COLORMAP_JET).astype(np.float32)

    y_flow_colormap = cv2.applyColorMap(
        (255 * scaled_y_flow).astype(np.uint8),
        cv2.COLORMAP_TWILIGHT).astype(np.float32)

    combined_colormap = np.clip((x_flow_colormap + y_flow_colormap) / 2, 0, 255)

    clipped_combined_weights_3d = np.repeat(scaled_combined_flow[:, :, np.newaxis], 3, axis=2)

    # Overlay weighted flow colormap over original image
    overlay = (image * (1 - clipped_combined_weights_3d)) + (combined_colormap * clipped_combined_weights_3d)

    return np.clip(overlay, 0, 255)
