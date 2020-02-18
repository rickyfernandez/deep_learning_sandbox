import re
import random
from pathlib import Path
from typing import Iterable, Any

import torch
from torch import optim
import torch.nn.functional as F

from torch.utils.data import DataLoader, SequentialSampler, RandomSampler

from .processor import Processor
from .utils import uniqueify, listify, compose

def normalize(x, m, s):
    return (x-m)/s


def normalize_to(train, valid):
    m, s = train.mean(), train.std()
    return normalize(train, m, s), normalize(valid, m, s)


class Dataset:
    def __init__(self, x, y): self.x, self.y = x, y
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]


def get_dls(train_ds, valid_ds, bs, **kwargs):
    return (DataLoader(train_ds, batch_size=bs, shuffle=True, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, **kwargs))

class DataBunch:
    def __init__(self, train_dl, valid_dl, c=None):
        self.train_dl, self.valid_dl, self.c = train_dl, valid_dl, c

    @property
    def train_ds(self): return self.train_dl.dataset

    @property
    def valid_ds(self): return self.valid_dl.dataset


class CategoryProcessor(Processor):
    def __init__(self): self.vocab = None

    def __call__(self, items):
        # The vocab is defined on the first use
        if self.vocab is None:
            self.vocab = uniqueify(items)
            self.otoi = {v:k for k,v in enumerate(self.vocab)}
        return [self.proc1(o) for o in items]

    def proc1(self, item): return self.otoi[item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]

    def deproc1(self, idx): return self.vocab[idx]



def parent_labeler(fn): return fn.parent.name


def grandparent_splitter(fn, valid_name="valid", train_name="train"):
    gp = fn.parent.parent.name
    return True if gp == valid_name else False if gp==train_name else None


def random_splitter(item, p_valid): return random.random() < p_valid


def split_by_func(itemlist, func):
    """Split data by using a function to create a bool array to indicate
    what partition that data will be put in ."""
    mask = [func(item) for item in itemlist]
    # None values will be filtered out
    items_fal = [item for item,flag in zip(itemlist, mask) if flag==False]
    items_tru = [item for item,flag in zip(itemlist, mask) if flag==True ]
    return items_fal, items_tru


class SplitData:
    """Class that holds train and valid. It also performs spitting if
    a validation set does not exist by using split_by_func.
    """
    def __init__(self, train, valid):
        self.train, self.valid = train, valid
    def __getattr__(self, k): return getattr(self.train, k)
    def __setstate__(self, data:Any): self.__dict__.update(data)

    @classmethod
    def split_by_func(cls, itemlist, func):
        """Split list of data into a training and validation data set by
        using func."""
        lists = map(itemlist.new, split_by_func(itemlist.items, func))
        return cls(*lists)

    def __repr__(self):
        return f'{self.__class__.__name__}\nTrain: {self.train}\n\nValid: {self.valid}\n'


class ListContainer:
    """A more useful form of python list, internal items are placed
    in a list. Indexing can index value, slice, booling array (has to
    be of same lenght), or index array."""
    def __init__(self, items): self.items = listify(items)
    def __getitem__(self, idx):
        # torch can use 0-dim tensors
        if isinstance(idx, torch.Tensor) and idx.shape == torch.Size([]):
            return self.items[idx]
        if isinstance(idx, (int, slice)): return self.items[idx]
        if isinstance(idx[0], bool):
            assert len(idx)==len(self)
            return [o for m,o in zip(idx, self.items) if m]
        return [self.items[i] for i in idx]
    def __len__(self): return len(self.items)
    def __iter__(self): return iter(self.items)
    def __setitem__(self, i, o): self.items[i] = o
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self):
        res = f'{self.__class__.__name__} ({len(self)} items)\n{self.items[:10]}'
        if len(self)>10: res = res[:-1] + '...]'
        return res


class ItemList(ListContainer):
    """Class that holds a list of items with wrappers to transfrom the items
    when retrieved. Transforms are a list of composition functions.
    """
    def __init__(self, items, path=".", transforms=None):
        super().__init__(items)
        self.path, self.transforms = Path(path), transforms

    def __repr__(self): return f'{super().__repr__()}\nPath: {self.path}'

    def new(self, items, cls=None):
        if cls is None: cls=self.__class__
        return cls(items, self.path, transforms=self.transforms)

    def get(self, i): return i
    def _get(self, i): return compose(self.get(i), self.transforms)

    def __getitem__(self, idx):
        res = super().__getitem__(idx)
        if isinstance(res, list): return [self._get(o) for o in res]
        return self._get(res)


def _label_by_func(ds, f, cls=ItemList):
    return cls([f(o) for o in ds.items], path=ds.path)


def label_by_func(sd, f, proc_x=None, proc_y=None):
    train = LabeledData.label_by_func(sd.train, f, proc_x=proc_x, proc_y=proc_y)
    valid = LabeledData.label_by_func(sd.valid, f, proc_x=proc_x, proc_y=proc_y)
    return SplitData(train, valid)


class LabeledData:
    def __init__(self, x, y, proc_x=None, proc_y=None):
        self.x, self.y = self.process(x, proc_x), self.process(y, proc_y)
        self.proc_x, self.proc_y = proc_x, proc_y

    def __repr__(self): return f"{self.__class__.__name__}\nx: {self.x}\ny: {self.y}\n"
    def __getitem__(self, idx): return self.x[idx], self.y[idx]
    def __len__(self): return len(self.x)

    def process(self, il, proc): return il.new(compose(il.items, proc))
    def x_obj(self, idx): return self.obj(self.x, idx, self.proc_x)
    def y_obj(self, idx): return self.obj(self.y, idx, self.proc_y)

    def obj(self, items, idx, procs):
        isint = isinstance(idx, int) or (isinstance(idx, torch.LongTensor) and not idx.ndim)
        item = items[idx]
        for proc in reversed(listify(procs)):
            item = proc.decode(item) if isint else proc.decode_batch(item)
        return item

    @classmethod
    def label_by_func(cls, il, f, proc_x=None, proc_y=None):
        return cls(il, _label_by_func(il, f), proc_x=proc_x, proc_y=proc_y)



#_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
#_camel_re2 = re.compile('([a-z0-9])([A-Z])')
#def camel2snake(name):
#    s1 = re.sub(_camel_re1, r'\1_\2', name)
#    return re.sub(_camel_re2, r'\1_\2', s1).lower()
#
#
#from typing import *
#
#
#def listify(obj):
#    if obj is None: return []
#    if isinstance(obj, list): return obj
#    if isinstance(obj, str): return [obj]
#    if isinstance(obj, Iterable): return list(obj)
#    return [obj]
#
#
#class AvgStats():
#    def __init__(self, metrics, in_train): self.metrics,self.in_train = listify(metrics),in_train
#
#    def reset(self):
#        self.tot_loss,self.count = 0.,0
#        self.tot_mets = [0.] * len(self.metrics)
#
#    @property
#    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets
#    @property
#    def avg_stats(self): return [o/self.count for o in self.all_stats]
#
#    def __repr__(self):
#        if not self.count: return ""
#        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"
#
#    def accumulate(self, run):
#        bn = run.xb.shape[0]
#        self.tot_loss += run.loss * bn
#        self.count += bn
#        for i,m in enumerate(self.metrics):
#            self.tot_mets[i] += m(run.pred, run.yb) * bn
#
#class AvgStatsCallback(Callback):
#    def __init__(self, metrics):
#        self.train_stats,self.valid_stats = AvgStats(metrics,True),AvgStats(metrics,False)
#
#    def begin_epoch(self):
#        self.train_stats.reset()
#        self.valid_stats.reset()
#
#    def after_loss(self):
#        stats = self.train_stats if self.in_train else self.valid_stats
#        with torch.no_grad(): stats.accumulate(self.run)
#
#    def after_epoch(self):
#        print(self.train_stats)
#        print(self.valid_stats)
#
#from functools import partial
