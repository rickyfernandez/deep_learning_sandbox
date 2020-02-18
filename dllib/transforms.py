
class Transform:
    _order=0


class ToFloatTensor(Transform):
    _order=100
    def __call__(self, item):
        return item.float().div_(255)


