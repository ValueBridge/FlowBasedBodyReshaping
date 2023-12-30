"""
Module with machine learning code
"""

import cv2
import numpy as np
import torch

import config.test_config
import network.flow_generator
import pose_estimator.body
import reshape_base_algos.image_warp
import reshape_base_algos.person_info
import reshape_base_algos.slim_utils


class BodyRetoucher:
    """
    Class with logic for retouching bodies
    """


    def __init__(
            self,
            reshape_ckpt_path,
            pose_estimation_ckpt,
            device,
            debug_level=0,
            network_input_H = 256,
            network_input_W = 256):

        self.debug_level = debug_level
        self.pad_border = True

        self.warp_lib = None

        self.device = torch.device("cuda:{}".format(device) if torch.cuda.is_available() else 'cpu')

        self.pose_estimator = pose_estimator.body.Body(pose_estimation_ckpt)

        self.network_input_H = network_input_H
        self.network_input_W = network_input_W

        self.flow_generator = network.flow_generator.FlowGenerator(n_channels=16)
        self.flow_generator.name = "FlowGenerator"

        self.liquify_net = self.flow_generator.to(self.device)

        checkpoint = torch.load(reshape_ckpt_path, map_location=self.device)

        self.flow_generator.load_state_dict(checkpoint['state_dict'], strict=True)
        self.flow_generator.eval()

    def predict_joints(self, img):

        small_src, resize_scale = reshape_base_algos.slim_utils.resize_on_long_side(img, 300)
        body_joints = self.pose_estimator(small_src)

        if body_joints.shape[0] >= 1:
            body_joints[:, :, :2] = body_joints[:, :, :2] / resize_scale

        return body_joints

    def predict_flow(self, img, resized_image, person: reshape_base_algos.person_info.PersonInfo):

        with torch.no_grad():
            person_pred = person.pred_flow(resized_image, self.flow_generator,  self.device)

        flow = np.dstack((person_pred['rDx'], person_pred['rDy']))

        scale = img.shape[0] *1.0/ flow.shape[0]

        flow = cv2.resize(flow, (img.shape[1], img.shape[0]))
        flow *= scale

        return flow

    def warp(self, src_img, flow):

        assert src_img.shape[:2] == flow.shape[:2]
        X_flow = flow[..., 0]
        Y_flow = flow[..., 1]

        X_flow = np.ascontiguousarray(X_flow)
        Y_flow = np.ascontiguousarray(Y_flow)

        pred = reshape_base_algos.image_warp.image_warp_grid1(X_flow, Y_flow, src_img, 1.0, 0, 0)

        return pred

    def reshape_body(
            self, src_img, scaled_down_image, person_info: reshape_base_algos.person_info.PersonInfo,
            degree=1.0):

        flow = self.predict_flow(src_img, scaled_down_image, person_info)

        if config.test_config.TESTCONFIG.suppress_bg:

            mag, ang = cv2.cartToPolar(flow[..., 0] + 1e-8, flow[..., 1] + 1e-8)
            mag -= 3
            mag[mag <= 0] = 0

            x, y = cv2.polarToCart(mag, ang, angleInDegrees=False)
            flow = np.dstack((x, y))

        flow *= degree

        prediction = self.warp(src_img, flow)
        prediction = np.clip(prediction, 0, 255)

        return prediction, flow
