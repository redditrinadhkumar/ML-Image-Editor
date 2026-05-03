import numpy as np
import cv2
from PIL import Image
import io


def pil_to_numpy(pil_img: Image.Image) -> np.ndarray:
    rgb_array = np.array(pil_img)
    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)


def numpy_to_pil(img_array: np.ndarray) -> Image.Image:
    if len(img_array.shape) == 2:
        return Image.fromarray(img_array)
    rgb_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_array)


def pil_to_bytes(pil_img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return buf.getvalue()


def numpy_to_bytes(img_array: np.ndarray, fmt: str = "PNG") -> bytes:
    return pil_to_bytes(numpy_to_pil(img_array), fmt)


def image_info(img_array: np.ndarray) -> dict:
    ndim = img_array.ndim
    shape = img_array.shape
    if ndim == 2:
        mode = "Grayscale (L)"
    elif ndim == 3 and shape[2] == 3:
        mode = "Color (BGR)"
    elif ndim == 3 and shape[2] == 4:
        mode = "Color+Alpha (BGRA)"
    else:
        mode = "Unknown"
    size_kb = round((img_array.nbytes) / 1024, 1)
    return {
        "shape": shape,
        "ndim": ndim,
        "mode": mode,
        "dtype": str(img_array.dtype),
        "size_kb": size_kb,
        "width": shape[1],
        "height": shape[0],
    }