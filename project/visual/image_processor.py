from PIL import Image
from io import BytesIO


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png"
}


def validate_image(content_type: str):
    """
    Check whether uploaded file is a supported image.
    """

    if content_type not in ALLOWED_IMAGE_TYPES:
        raise ValueError(
            "Unsupported image format. "
            "Only JPG, JPEG and PNG are allowed."
        )


def load_image(image_bytes: bytes) -> Image.Image:
    """
    Convert uploaded bytes into a PIL Image.
    """

    try:
        image_stream = BytesIO(image_bytes)
        image = Image.open(image_stream)
        image.verify()

        # Reopen after verify because verify() leaves the image unusable.
        image = Image.open(BytesIO(image_bytes))

        # Convert to RGB because Gemma expects RGB images
        image = image.convert("RGB")

        return image

    except Exception as e:
        raise ValueError(f"Invalid image: {str(e)}")