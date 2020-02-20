import PIL
from ..data import ItemList
from ..utils import get_files


class ImageList(ItemList):
    @classmethod
    def from_files(cls, path, extensions='.png', recurse=True, include=None, **kwargs):
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, filename):
        return PIL.Image.open(filename)
