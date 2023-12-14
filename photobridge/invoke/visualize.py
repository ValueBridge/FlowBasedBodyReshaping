"""
Module with visualization commands
"""

import invoke


@invoke.task
def visualize_body_reshaping(_context, config_path):
    """
    Visualize mesh predictions

    Args:
        _context (invoke.Context): context instance
        config_path (str): path to configuration file
    """

    import glob
    import os

    import cv2
    import icecream
    import shutil
    import toml
    import tqdm

    import config.test_config
    import reshape_base_algos.body_retoucher


    import photobridge.utilities

    # Delete all images in tmp directory
    for path in glob.glob("/tmp/*.jpg"):
        os.remove(path)

    configuration = photobridge.utilities.read_yaml(config_path)

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
