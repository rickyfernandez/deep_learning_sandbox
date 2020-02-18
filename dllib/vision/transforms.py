import PIL
import torch
from ..transforms import Transform


class MakeRGB(Transform):
    """
    Transform image into RGB format.
    """
    _order=0
    def __call__(self, item):
        return item.convert('RGB')

class ToByteTensor(Transform):
    """
    Transform image to torch tensor and make image
    channel first, height, width to be consistent
    with pytorch.
    """
    _order=30
    def __call__(self, item):
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
        w, h = item.size
        return res.view(h, w, -1).permute(2, 0, 1)

class ResizeFixed(Transform):
    """
    Transform image to desired height and width.
    """
    _order=10
    def __init__(self, height_width, interpolation=PIL.Image.BILINEAR):
        if isinstance(height_width, int):
            height_width = (height_width, height_width)
        self.height_width = height_width
        self.interpolation = interpolation

    def __call__(self, item):
        return item.resize(self.height_width, self.interpolation) 

