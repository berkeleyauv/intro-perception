from __future__ import annotations
 
from pathlib import Path
 
import cv2 as cv
 
 
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
 
 
def list_images(path):
    """Return a sorted list of image files for ``path``.
 
    ``path`` may be a single image or a folder (searched non-recursively).
    Raises ``ValueError`` for videos or other file types, for a folder with no
    images, and when two images share a name (``a.jpg`` and ``a.png``), because
    the file name without its extension is the key used for labels and
    predictions everywhere else.
    """
    path = Path(path)
    if path.is_dir():
        images = sorted(
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise ValueError(
                f"no images ({', '.join(IMAGE_SUFFIXES)}) found in {path}"
            )
    elif path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(
                f"{path}: only still images ({', '.join(IMAGE_SUFFIXES)}) are "
                "supported; videos are not"
            )
        images = [path]
    else:
        raise ValueError(f"{path} does not exist")
 
    seen = {}
    for image in images:
        if image.stem in seen:
            raise ValueError(
                f"{seen[image.stem].name} and {image.name} share the name "
                f"{image.stem!r}; rename one so labels and predictions stay unique"
            )
        seen[image.stem] = image
    return images
 
 
def read_image(path):
    """Load one image as a BGR array, raising ``ValueError`` if it is unreadable."""
    frame = cv.imread(str(path))
    if frame is None:
        raise ValueError(f"could not read image: {path}")
    return frame
