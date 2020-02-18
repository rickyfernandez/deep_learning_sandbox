import PIL
from ..data import ItemList


class ImageList(ItemList):
    def get(self, filename):
        return PIL.Image.open(filename)
