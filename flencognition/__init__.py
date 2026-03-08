"""
FLENcognition – document-image OCR toolkit powered by FireRed-OCR.

Quick-start
-----------
>>> from flencognition import FLENcognition
>>> ocr = FLENcognition()
>>> result = ocr.process_image("page.png")
>>> print(result["markdown"])

Or use the module-level convenience functions for a zero-configuration
experience::

    from flencognition import process_image, process_images
    result  = process_image("page.png")
    results = process_images(["p1.png", "p2.png"])
"""

from .core import FLENcognition

__all__ = [
    "FLENcognition",
    "process_image",
    "process_images",
]

# ---------------------------------------------------------------------------
# Module-level convenience helpers backed by a lazily-created default engine
# ---------------------------------------------------------------------------

_default_engine: FLENcognition | None = None


def _get_default_engine() -> FLENcognition:
    global _default_engine
    if _default_engine is None:
        _default_engine = FLENcognition()
    return _default_engine


def process_image(image_path: str, **kwargs) -> dict:
    """Process a single document image using the default engine.

    This is a convenience wrapper around :meth:`FLENcognition.process_image`.
    The underlying model is loaded on first call.

    Parameters
    ----------
    image_path:
        Path to the image file.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`FLENcognition.process_image` (e.g. ``save_markdown=True``).

    Returns
    -------
    dict
        ``{"markdown": str, "latex": str, "file": str | None}``
    """
    return _get_default_engine().process_image(image_path, **kwargs)


def process_images(image_paths: list[str], **kwargs) -> list[dict]:
    """Process multiple document images using the default engine.

    This is a convenience wrapper around :meth:`FLENcognition.process_images`.
    The underlying model is loaded on first call.

    Parameters
    ----------
    image_paths:
        List of paths to image files.
    **kwargs:
        Additional keyword arguments forwarded to
        :meth:`FLENcognition.process_images` (e.g. ``save_markdown=True``).

    Returns
    -------
    list[dict]
        One result dict per image.
    """
    return _get_default_engine().process_images(image_paths, **kwargs)
