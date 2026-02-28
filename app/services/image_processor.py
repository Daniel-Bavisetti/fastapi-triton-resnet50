from io import BytesIO

import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageProcessingError(ValueError):
    """Raised when image bytes cannot be decoded or transformed."""


def preprocess_image(image_bytes: bytes, input_size: int = 224) -> np.ndarray:
    if not image_bytes:
        raise ImageProcessingError("Empty image payload")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image = image.convert("RGB")
            image = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
            image_np = np.array(image, dtype=np.float32) / 255.0
    except UnidentifiedImageError as exc:
        raise ImageProcessingError("Invalid image content") from exc

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_np = (image_np - mean) / std

    chw = np.transpose(image_np, (2, 0, 1))
    batched = np.expand_dims(chw, axis=0)
    batched = np.ascontiguousarray(batched, dtype=np.float32)
    if batched.shape != (1, 3, 224, 224):
        raise ImageProcessingError("Preprocessed tensor shape must be (1, 3, 224, 224)")
    return batched
