import math
import torch
import random
import os, mimetypes, re
from typing import Iterable, Any
from pathlib import Path
from collections import OrderedDict


_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')


def accuracy(out, yb):
    return (torch.argmax(out, dim=1)==yb).float().mean()


def camel2snake(name):
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()


def listify(obj):
    """Create list of obj by checking type and chaninging to list
    if not list, if none return empty list."""
    if obj is None: return []
    if isinstance(obj, list): return obj
    if isinstance(obj, str): return [obj]
    if isinstance(obj, Iterable): return list(obj)
    return [obj]


def compose(x, funcs, *args, order_key='_order', **kwargs):
    """Perform a serious transformations on x by composition i.e.
    x = (f o g o h)(x)."""
    key = lambda o: getattr(o, order_key, 0)
    for f in sorted(listify(funcs), key=key): x = f(x, **kwargs)
    return x


def setify(o): return o if isinstance(o, set) else set(listify(o))


def uniqueify(x, sort=False):
    res = list(OrderedDict.fromkeys(x).keys())
    if sort: res.sort()
    return res


def _get_files(p, fs, extensions=None):
    """Grab all files with path that have `extensions`."""
    p = Path(p)
    res = [p/f for f in fs if not f.startswith('.')
           and ((not extensions) or f'.{f.split(".")[-1].lower()}' in extensions)]
    return res


def get_files(path, extensions=None, recurse=False, include=None):
    """Grab all paths for files  with `extensions` in folders. Can subset lookup
    folders in first level by include argument. This can recursively go down and
    collect files with `recurse flag`. A gotcha is that walk is modified by changing
    dirname in place even though it is not used..
    """
    path = Path(path)
    extensions = setify(extensions)
    extensions = {e.lower() for e in extensions}
    if recurse:
        res = []
        for i, (dirpath, dirname, filename) in enumerate(os.walk(path)):
            if include is not None and i==0:
                # include folders of interest
                dirname[:] = [folder for folder in dirname if folder in include]
            else:
                # ignore hidden files
                dirname[:] = [folder for folder in dirname if not folder.startswith('.')]
            res += _get_files(dirpath, filename, extensions)
        return res
    else:
        # return list of absolute path to filenames with extensions
        filenames = [o.name for o in os.scandir(path) if o.is_file()]
        return _get_files(path, filenames, extensions)







def prev_pow_2(x): return 2**math.floor(math.log2(x))






def model_summary(run, learn, data, find_all=False):
    xb,yb = get_batch(data.valid_dl, run)
    device = next(learn.model.parameters()).device#Model may not be on the GPU yet
    xb,yb = xb.to(device),yb.to(device)
    mods = find_modules(learn.model, is_lin_layer) if find_all else learn.model.children()
    f = lambda hook,mod,inp,out: print(f"{mod}\n{out.shape}\n")
    with Hooks(mods, f) as hooks: learn.model(xb)
