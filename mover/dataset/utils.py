import numpy as np

# The group of method to scale the rgb image
def rgb_image_normalize(
        img,  # type: np.ndarray
        rgb_mean, # type: List[float],
        rgb_scale, # type: List[float],
        rgb_std # type: List[float],
):  # type: (np.ndarray, List[float], List[float], List[float]) -> np.ndarray
    """
    (height, width, channels) -> (channels, height, width), BGR->RGB and normalize
    :param img: The raw opencv image as np.ndarray in the shape of (height, width, 3)
    :param rgb_mean: The mean value for RGB, all of them are in [0, 1]
    :return: The normalized, randomized RGB tensor
    """
    tensor = img.copy()
    tensor = tensor.astype(np.float)

    # Scale and normalize
    normalizer = [1.0/255.0, 1.0/255.0, 1.0/255.0]
    for i in range(3):
        normalizer[i] = normalizer[i] * rgb_scale[i]

    # Apply to image
    for channel in range(len(rgb_mean)):
        tensor[:, :, channel] = ((normalizer[channel] * tensor[:, :, channel]) - rgb_mean[channel]) / rgb_std[channel]

    return tensor

def depth_image_normalize(
        depth,  # type: np.ndarray
        depth_clip,  # type: int
        depth_mean,  # type: int
        depth_scale,  # type: int
):  # type: (np.ndarray, int, int, int) -> np.ndarray
    """
    :param depth: image in the size of (img_height, img_width)
    :param depth_clip:
    :param depth_mean:
    :param depth_scale:
    :return: out = (clip(depth_in, 0, depth_clip) - depth_mean) / depth_scale
    """
    tensor = depth.copy()
    tensor[tensor >= depth_clip] = 0

    # Do normalize
    tensor = tensor.astype(np.float)
    tensor -= float(depth_mean)
    normalizer = 1.0 / float(depth_scale)
    tensor = tensor * normalizer
    return tensor