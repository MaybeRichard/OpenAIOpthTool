import numpy as np
from PIL import Image


def load_image(image):
    """
    将输入统一为 PIL RGB Image / Normalize any input to PIL RGB Image.

    Args:
        image: File path (str), PIL Image, or numpy ndarray (HWC, uint8, RGB).

    Returns:
        PIL.Image.Image in RGB mode.
    """
    if isinstance(image, str):
        return Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        return image.convert("RGB")
    elif isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        elif image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image, mode="RGB")
        else:
            raise ValueError(f"Unsupported ndarray shape: {image.shape}")
    else:
        raise TypeError(
            f"Unsupported image type: {type(image)}. "
            f"Expected str, PIL.Image, or numpy.ndarray."
        )
