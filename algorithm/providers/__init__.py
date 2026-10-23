from .base import ImageRepairProvider, NullProvider
from .gemini_image import GeminiImageProvider, create_provider


def create_repair_provider(kind: str):
    return create_provider(kind)
