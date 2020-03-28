import PIL
import math
import torch
import torch.nn.functional as F
from torch import stack, zeros_like, ones_like
from ..transforms import Transform


class MakeRGB(Transform):
    """Transform image into RGB format."""
    _order=0
    def __call__(self, item):
        return item.convert('RGB')

class ToByteTensor(Transform):
    """Transform image to torch tensor and make image
    channel first, height, width to be consistent
    with pytorch."""
    _order=30
    def __call__(self, item):
        res = torch.ByteTensor(torch.ByteStorage.from_buffer(item.tobytes()))
        w, h = item.size
        return res.view(h, w, -1).permute(2, 0, 1)

class ResizeFixed(Transform):
    """Transform image to desired height and width."""
    _order=10
    def __init__(self, height_width, interpolation=PIL.Image.BILINEAR):
        if isinstance(height_width, int):
            height_width = (height_width, height_width)
        self.height_width = height_width
        self.interpolation = interpolation

    def __call__(self, item):
        return item.resize(self.height_width, self.interpolation) 

def affine_matrix(*vs):
    """Create affine matrix from 6 1-dim vectors. Rows of the
    vector aleternate between x and y transforms for each batch
    value."""
    return stack([stack([vs[0], vs[1], vs[2]], dim=1),
                  stack([vs[3], vs[4], vs[5]], dim=1)], dim=1)


def create_identity_affine(x):
    """Create an identity matrix for affine transformation.
    Since affine_grid expects transforms in the form 2x3 matrix
    we slice to the correct dimensions. This affine transform
    does not alter the data."""
    batch_size = x.size(0)
    eye = torch.eye(3, device=x.device).float()
    eye = eye.expand(batch_size, 3, 3)[:,:2]
    return eye

def affine_transform(x, mat_trans=None, grid_transform=None, mode="bilinear",
        padding_mode="reflection", align_corners=True):
    """Perform an affine transformation on a batch of images. If a
    matrix transformation is provided then we use it to pefrom the transformation
    otherwise we create an indentity transform and use grid_transform."""
    if mat_trans is None: mat_trans = create_identity_affine(x)
    grid = F.affine_grid(mat_trans, x.size(), align_corners=align_corners)
    if grid_transform is not None: grid = grid_transform(grid)
    return F.grid_sample(x, grid, mode=mode, padding_mode=padding_mode,
            align_corners=align_corners)

def rotation_matrix(x, degrees=10):
    """Create a rotation matrix for image in batch. Each rotation matrix
    has a random rotation uniformly between (-degrees, degrees)."""
    batch_size = x.size(0)
    thetas = x.new(batch_size).uniform_(-degrees, degrees).mul_(math.pi/180)
    matrix = affine_matrix( thetas.cos(), thetas.sin(), zeros_like(thetas),
                           -thetas.sin(), thetas.cos(), zeros_like(thetas))
    return matrix
         
def horizontal_flip_matrix(x, p=0.5):
    """Create a random horizontal flip matrix for image in batch. Each flip
    matrix has a random chance of flipping with value `p`."""
    batch_size = x.size(0)
    mask = x.new_empty(batch_size).bernoulli_(p)
    mask = x.new_ones(batch_size) - 2*mask
    return affine_matrix(mask,            zeros_like(mask), zeros_like(mask),
                         zeros_like(mask), ones_like(mask), zeros_like(mask))

def vertical_flip_matrix(x, p=0.5):
    """Create a random vertical flip matrix for image in batch. Each flip
    matrix has a random chance of flipping with value `p`."""
    batch_size = x.size(0)
    mask = x.new_empty(batch_size).bernoulli_(p)
    mask = x.new_ones(batch_size) - 2*mask
    return affine_matrix(ones_like(mask), zeros_like(mask), zeros_like(mask),
                         zeros_like(mask),            mask, zeros_like(mask))
