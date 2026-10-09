"""Image loading and validation module."""

import hashlib
from pathlib import Path
import PIL.Image
import PIL.ImageOps


def load_image(path: str | Path) -> PIL.Image.Image:
    """Load and validate an image file.
    
    Accepts PNG, JPEG, DICOM-exported PNG.
    Validates with Pillow, converts to RGB.
    Preserves EXIF orientation.
    
    Args:
        path: Path to the image file.
        
    Returns:
        The loaded PIL Image object.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is corrupt, unsupported, or has zero dimensions.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
        
    try:
        # Load image and convert to RGB
        img = PIL.Image.open(path)
        img.verify()  # verify format is supported and file is not completely corrupt
        
        # Must reopen after verify()
        img = PIL.Image.open(path)
        img = PIL.ImageOps.exif_transpose(img) # Preserve EXIF orientation
        img = img.convert("RGB")
        
        # Check dimensions
        if img.width == 0 or img.height == 0:
            raise ValueError(f"Image has zero dimensions: {img.size}")
            
        return img
    except (PIL.Image.UnidentifiedImageError, OSError) as e:
        raise ValueError(f"Unsupported or corrupt image format: {path}") from e


def image_sha256(path: str | Path) -> str:
    """Compute the SHA256 hash of an image file.
    
    Args:
        path: Path to the image file.
        
    Returns:
        Hexadecimal SHA256 hash string.
    """
    path = Path(path)
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
