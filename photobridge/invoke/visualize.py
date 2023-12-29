"""
Module with visualization commands
"""

import invoke


@invoke.task
def body_reshaping(_context, config_path):
    """
    Visualize mesh predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import glob
    import os

    import cv2
    import toml
    import tqdm

    import config.test_config
    import reshape_base_algos.body_retoucher

    import photobridge.utilities

    configuration = photobridge.utilities.read_yaml(config_path)

    photobridge.utilities.delete_files(
        directory=configuration.output_images_directory,
        extensions=["jpg", "gif", "mp4"]
    )

    reshaper_config_path = "config/test_cvpr_setting.toml"
    reshaper_default_config = config.test_config.TESTCONFIG

    with open(reshaper_config_path) as file:

        reshaper_config = toml.load(file)

        reshaper_config = config.test_config.load_config(
            custom_config=reshaper_config,
            default_config=reshaper_default_config)

    reshape_base_algos.body_retoucher.BodyRetoucher.init(
        reshape_ckpt_path=reshaper_default_config.reshape_ckpt_path,
        pose_estimation_ckpt=reshaper_default_config.pose_estimation_ckpt,
        device=0, log_level='error',
        log_path='test_log.txt',
        debug_level=0)

    image_paths = glob.glob(os.path.join(configuration.input_images_directory, "*.jpg"))

    for image_path in tqdm.tqdm(image_paths):

        source_image = cv2.imread(image_path)

        prediction, flow = reshape_base_algos.body_retoucher.BodyRetoucher.reshape_body(
            src_img=source_image,
            degree=config.test_config.TESTCONFIG.degree)

        file_stem = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(
            os.path.join(configuration.output_images_directory, f"{file_stem}_a_source.jpg"),
            source_image
        )

        cv2.imwrite(
            os.path.join(configuration.output_images_directory, f"{file_stem}_b_prediction.jpg"),
            prediction
        )


@invoke.task
def body_reshaping_animations(_context, config_path):
    """
    Read in images from directory specified in configuration,
    and for each image create a gif animation showing varying degrees of body reshaping

    Args:
        _context (invoke.Context): invoke.Context instance
        config_path (str): path to configuration file
    """

    import glob
    import os

    import cv2
    import moviepy.editor
    import numpy as np
    import toml
    import tqdm

    import config.test_config
    import reshape_base_algos.body_retoucher

    import photobridge.utilities

    configuration = photobridge.utilities.read_yaml(config_path)

    photobridge.utilities.delete_files(
        directory=configuration.output_images_directory,
        extensions=["jpg", "gif", "mp4"]
    )

    reshaper_config_path = "config/test_cvpr_setting.toml"
    reshaper_default_config = config.test_config.TESTCONFIG

    with open(reshaper_config_path) as file:

        reshaper_config = toml.load(file)

        reshaper_config = config.test_config.load_config(
            custom_config=reshaper_config,
            default_config=reshaper_default_config)

    reshape_base_algos.body_retoucher.BodyRetoucher.init(
        reshape_ckpt_path=reshaper_default_config.reshape_ckpt_path,
        pose_estimation_ckpt=reshaper_default_config.pose_estimation_ckpt,
        device=0, log_level='error',
        log_path='test_log.txt',
        debug_level=0)

    image_paths = sorted(glob.glob(os.path.join(configuration.input_images_directory, "*.jpg")))

    for image_path in tqdm.tqdm(image_paths):

        source_image = cv2.imread(image_path)

        frames = [source_image]

        for degree in np.arange(0.0, 2, 0.25):

            prediction, _ = reshape_base_algos.body_retoucher.BodyRetoucher.reshape_body(
                src_img=source_image,
                degree=degree)

            frames.append(prediction)

        # Make a video from frames with moviepy
        def make_frame(time):

            time = int(time)
            return cv2.cvtColor(frames[time], cv2.COLOR_BGR2RGB)

        moviepy.editor.VideoClip(make_frame, duration=len(frames) - 1).write_videofile(
            filename=os.path.join(
                configuration.output_images_directory,
                f"{os.path.splitext(os.path.basename(image_path))[0]}.mp4"),
            fps=2,
            audio=False
        )


@invoke.task
def pose_estimations(_context, config_path):
    """
    Visualize pose estimations

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import glob
    import os

    import cv2
    import icecream
    import numpy as np
    import toml
    import tqdm

    import config.test_config
    import pose_estimator.body
    import reshape_base_algos.body_retoucher
    import reshape_base_algos.slim_utils

    import photobridge.logging
    import photobridge.utilities

    # Suppress scientific notation in numpy
    np.set_printoptions(suppress=True)

    configuration = photobridge.utilities.read_yaml(config_path)

    photobridge.utilities.delete_files(
        directory=configuration.output_images_directory,
        extensions=["jpg", "gif", "mp4"]
    )

    reshaper_default_config = config.test_config.TESTCONFIG

    pose_estimation_model = pose_estimator.body.Body(reshaper_default_config.pose_estimation_ckpt)

    image_paths = sorted(glob.glob(os.path.join(configuration.input_images_directory, "*.jpg")))

    for image_path in tqdm.tqdm(image_paths):

        image = cv2.imread(image_path)

        resized_image, resize_scale = reshape_base_algos.slim_utils.resize_on_long_side(
            img=image,
            long_side=300)

        body_joints = pose_estimation_model(resized_image)

        if body_joints.shape[0] >= 1:
            body_joints[:, :, :2] = body_joints[:, :, :2] / resize_scale

        for person_body_joints in body_joints:

            image = photobridge.logging.get_pose_estimation_visualization(
                image=image,
                joints=person_body_joints)

        cv2.imwrite(
            os.path.join(
                configuration.output_images_directory,
                f"{os.path.splitext(os.path.basename(image_path))[0]}.jpg"),
            image
        )


@invoke.task
def flow_predictions(_context, config_path):
    """
    Visualize flow predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import glob
    import os

    import cv2
    import icecream
    import numpy as np
    import toml
    import tqdm

    import config.test_config
    import pose_estimator.body
    import reshape_base_algos.body_retoucher
    import reshape_base_algos.slim_utils

    import photobridge.logging
    import photobridge.utilities

    # Suppress scientific notation in numpy
    np.set_printoptions(suppress=True)

    configuration = photobridge.utilities.read_yaml(config_path)

    photobridge.utilities.delete_files(
        directory=configuration.output_images_directory,
        extensions=["jpg", "gif", "mp4"]
    )

    reshaper_config_path = "config/test_cvpr_setting.toml"
    reshaper_default_config = config.test_config.TESTCONFIG

    with open(reshaper_config_path) as file:

        reshaper_config = toml.load(file)

        reshaper_config = config.test_config.load_config(
            custom_config=reshaper_config,
            default_config=reshaper_default_config)

    reshape_base_algos.body_retoucher.BodyRetoucher.init(
        reshape_ckpt_path=reshaper_default_config.reshape_ckpt_path,
        pose_estimation_ckpt=reshaper_default_config.pose_estimation_ckpt,
        device=0, log_level='error',
        log_path='test_log.txt',
        debug_level=0)

    image_paths = sorted(glob.glob(os.path.join(configuration.input_images_directory, "*.jpg")))

    for image_path in tqdm.tqdm(image_paths):

        image = cv2.imread(image_path)

        _, flow = reshape_base_algos.body_retoucher.BodyRetoucher.reshape_body(
            src_img=image,
            degree=1.0)

        image_name = os.path.splitext(os.path.basename(image_path))[0]

        cv2.imwrite(
            os.path.join(
                configuration.output_images_directory,
                f"{image_name}_combined_flow.jpg"),
            photobridge.logging.get_flow_visualization_combined(image=image, flow=flow)
        )
