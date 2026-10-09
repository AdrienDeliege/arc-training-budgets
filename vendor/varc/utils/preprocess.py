from typing import List
from .arclib.augmenters import Augmenter, Rotate, Flip

def get_basic_augmenters() -> List[Augmenter]:
    return [Rotate(90), Rotate(270), Rotate(180), Flip(0), Flip(1)]
