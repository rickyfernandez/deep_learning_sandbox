import re, operator, sys, inspect
import random
from pathlib import Path
from typing import Iterable, Any
import itertools
from typing import Iterable, Generator
import numpy as np
from numpy import array,ndarray
from copy import copy
from functools import partial
from operator import itemgetter

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
    return (DataLoader(train_ds, batch_size=bs, shuffle=False, **kwargs),
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

class _Arg:
    def __init__(self, i): self.i = i

class bind:
    "Same as `partial`, except you can use `arg0` `arg1` etc param placeholders"
    def __init__(self, fn, *pargs, **pkwargs):
        self.fn, self.pargs, self.pkwargs = fn, pargs, pkwargs
        self.maxi = max((x.i for x in pargs if isinstance(x, _Arg)), default=-1)

    def __call__(self, *args, **kwargs):
        args = list(args)
        kwargs = {**self.pkwargs,**kwargs}
        for k,v in kwargs.items():
            if isinstance(v,_Arg): kwargs[k] = args.pop(v.i)
        fargs = [args[x.i] if isinstance(x, _Arg) else x for x in self.pargs] + args[self.maxi+1:]
        return self.fn(*fargs, **kwargs)


class FixSigMeta(type):
    "A metaclass that fixes the signature on classes that override __new__"
    def __new__(cls, name, bases, dict):
        res = super().__new__(cls, name, bases, dict)
        if res.__init__ is not object.__init__: res.__signature__ = inspect.signature(res.__init__)
        return res

class PrePostInitMeta(FixSigMeta):
    "A metaclass that calls optional `__pre_init__` and `__post_init__` methods"
    def __call__(cls, *args, **kwargs):
        res = cls.__new__(cls)
        if type(res)==cls:
            if hasattr(res,'__pre_init__'): res.__pre_init__(*args,**kwargs)
            res.__init__(*args,**kwargs)
            if hasattr(res,'__post_init__'): res.__post_init__(*args,**kwargs)
        return res

# Cell
class NewChkMeta(FixSigMeta):
    "Metaclass to avoid recreating object passed to constructor"
    def __call__(cls, x=None, *args, **kwargs):
        if not args and not kwargs and x is not None and isinstance(x,cls):
            x._newchk = 1
            return x

        res = super().__call__(*((x,) + args), **kwargs)
        res._newchk = 0
        return res

class CollBase:
    "Base class for composing a list of `items`"
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, k): return self.items[k]
    def __setitem__(self, k, v): self.items[list(k) if isinstance(k,CollBase) else k] = v
    def __delitem__(self, i): del(self.items[i])
    def __repr__(self): return self.items.__repr__()
    def __iter__(self): return self.items.__iter__()

def negate_func(f):
    "Create new function that negates result of `f`"
    def _f(*args, **kwargs): return not f(*args, **kwargs)
    return _f


def coll_repr(c, max_n=10):
    "String repr of up to `max_n` items of (possibly lazy) collection `c`"
    return f'(#{len(c)}) [' + ','.join(itertools.islice(map(repr,c), max_n)) + (
        '...' if len(c)>10 else '') + ']'

def zip_cycle(x, *args):
    "Like `itertools.zip_longest` but `cycle`s through elements of all but first argument"
    return zip(x, *map(cycle,args))

def is_iter(o):
    "Test whether `o` can be used in a `for` loop"
    #Rank 0 tensors in PyTorch are not really iterable
    return isinstance(o, (Iterable,Generator)) and getattr(o,'ndim',1)

def is_coll(o):
    "Test whether `o` is a collection (i.e. has a usable `len`)"
    #Rank 0 tensors in PyTorch do not have working `len`
    return hasattr(o, '__len__') and getattr(o,'ndim',1)

def noop (x=None, *args, **kwargs):
    "Do nothing"
    return x

def cycle(o):
    "Like `itertools.cycle` except creates list of `None`s if `o` is empty"
    o = _listify(o)
    return itertools.cycle(o) if o is not None and len(o) > 0 else itertools.cycle([None])

def _is_array(x): return hasattr(x,'__array__') or hasattr(x,'iloc')


def is_indexer(idx):
    "Test whether `idx` will index a single item in a list"
    return isinstance(idx,int) or not getattr(idx,'ndim',1)

def one_is_instance(a, b, t): return isinstance(a,t) or isinstance(b,t)

def equals(a,b):
    "Compares `a` and `b` for equality; supports sublists, tensors and arrays too"
    if one_is_instance(a,b,type): return a==b
    if hasattr(a, '__array_eq__'): return a.__array_eq__(b)
    if hasattr(b, '__array_eq__'): return b.__array_eq__(a)
    cmp = (np.array_equal if one_is_instance(a, b, ndarray       ) else
           operator.eq    if one_is_instance(a, b, (str,dict,set)) else
           all_equal      if is_iter(a) or is_iter(b) else
           operator.eq)
    return cmp(a,b)

NoneType = type(None)

def mask2idxs(mask):
    "Convert bool mask or index list to index `L`"
    if isinstance(mask,slice): return mask
    mask = list(mask)
    if len(mask)==0: return []
    it = mask[0]
    if hasattr(it,'item'): it = it.item()
    if isinstance(it,(bool,NoneType,np.bool_)): return [i for i,m in enumerate(mask) if m]
    return [int(i) for i in mask]

def all_equal(a,b):
    "Compares whether `a` and `b` are the same length and have the same contents"
    if not is_iter(b): return False
    return all(equals(a_,b_) for a_,b_ in itertools.zip_longest(a,b))

def _listify(o):
    if o is None: return []
    if isinstance(o, list): return o
    if isinstance(o, str) or _is_array(o): return [o]
    if is_iter(o): return list(o)
    return [o]

class L(CollBase, metaclass=NewChkMeta):
    "Behaves like a list of `items` but can also index with list of indices or masks"
    _default='items'
    def __init__(self, items=None, *rest, use_list=False, match=None):
        if rest: items = (items,)+rest
        if items is None: items = []
        if (use_list is not None) or not _is_array(items):
            items = list(items) if use_list else _listify(items)
        if match is not None:
            if is_coll(match): match = len(match)
            if len(items)==1: items = items*match
            else: assert len(items)==match, 'Match length mismatch'
        super().__init__(items)

    @property
    def _xtra(self): return None
    def _new(self, items, *args, **kwargs): return type(self)(items, *args, use_list=None, **kwargs)
    def __getitem__(self, idx): return self._get(idx) if is_indexer(idx) else L(self._get(idx), use_list=None)
    def copy(self): return self._new(self.items.copy())

    def _get(self, i):
        if is_indexer(i) or isinstance(i,slice): return getattr(self.items,'iloc',self.items)[i]
        i = mask2idxs(i)
        return (self.items.iloc[list(i)] if hasattr(self.items,'iloc')
                else self.items.__array__()[(i,)] if hasattr(self.items,'__array__')
                else [self.items[i_] for i_ in i])

    def __setitem__(self, idx, o):
        "Set `idx` (can be list of indices, or mask, or int) items to `o` (which is broadcast if not iterable)"
        idx = idx if isinstance(idx,L) else _listify(idx)
        if not is_iter(o): o = [o]*len(idx)
        for i,o_ in zip(idx,o): self.items[i] = o_

    def __iter__(self): return iter(self.items.itertuples() if hasattr(self.items,'iloc') else self.items)
    def __contains__(self,b): return b in self.items
    def __invert__(self): return self._new(not i for i in self)
    def __eq__(self,b): return False if isinstance(b, (str,dict,set)) else all_equal(b,self)
    def __repr__(self): return repr(self.items) if _is_array(self.items) else coll_repr(self)
    def __mul__ (a,b): return a._new(a.items*b)
    def __add__ (a,b): return a._new(a.items+_listify(b))
    def __radd__(a,b): return a._new(b)+a
    def __addi__(a,b):
        a.items += list(b)
        return a

    def sorted(self, key=None, reverse=False):
        if isinstance(key,str):   k=lambda o:getattr(o,key,0)
        elif isinstance(key,int): k=itemgetter(key)
        else: k=key
        return self._new(sorted(self.items, key=k, reverse=reverse))

    @classmethod
    def split(cls, s, sep=None, maxsplit=-1): return cls(s.split(sep,maxsplit))

    @classmethod
    def range(cls, a, b=None, step=None):
        if is_coll(a): a = len(a)
        return cls(range(a,b,step) if step is not None else range(a,b) if b is not None else range(a))

    def map(self, f, *args, **kwargs):
        g = (bind(f,*args,**kwargs) if callable(f)
             else f.format if isinstance(f,str)
             else f.__getitem__)
        return self._new(map(g, self))

    def filter(self, f, negate=False, **kwargs):
        if kwargs: f = partial(f,**kwargs)
        if negate: f = negate_func(f)
        return self._new(filter(f, self))

    def argwhere(self, f, negate=False, **kwargs):
        if kwargs: f = partial(f,**kwargs)
        if negate: f = negate_func(f)
        return self._new(i for i,o in enumerate(self) if f(o))

    def unique(self): return L(dict.fromkeys(self).keys())
    def enumerate(self): return L(enumerate(self))
    def val2idx(self): return {v:k for k,v in self.enumerate()}
    def itemgot(self, *idxs):
        x = self
        for idx in idxs: x = x.map(itemgetter(idx))
        return x

    def attrgot(self, k, default=None): return self.map(lambda o:getattr(o,k,default))
    def cycle(self): return cycle(self)
    def map_dict(self, f=noop, *args, **kwargs): return {k:f(k, *args,**kwargs) for k in self}
    def starmap(self, f, *args, **kwargs): return self._new(itertools.starmap(partial(f,*args,**kwargs), self))
    def zip(self, cycled=False): return self._new((zip_cycle if cycled else zip)(*self))
    def zipwith(self, *rest, cycled=False): return self._new([self, *rest]).zip(cycled=cycled)
    def map_zip(self, f, *args, cycled=False, **kwargs): return self.zip(cycled=cycled).starmap(f, *args, **kwargs)
    def map_zipwith(self, f, *rest, cycled=False, **kwargs): return self.zipwith(*rest, cycled=cycled).starmap(f, **kwargs)
    def concat(self): return self._new(itertools.chain.from_iterable(self.map(L)))
    def shuffle(self):
        it = copy(self.items)
        random.shuffle(it)
        return self._new(it)

    def append(self,o): return self.items.append(o)
    def remove(self,o): return self.items.remove(o)
    def count (self,o): return self.items.count(o)
    def reverse(self ): return self.items.reverse()
    def pop(self,o=-1): return self.items.pop(o)
    def clear(self   ): return self.items.clear()
    def index(self, value, start=0, stop=sys.maxsize): return self.items.index(value, start=start, stop=stop)
    def sort(self, key=None, reverse=False): return self.items.sort(key=key, reverse=reverse)
    def reduce(self, f, initial=None): return reduce(f, self) if initial is None else reduce(f, self, initial)
    def sum(self): return self.reduce(operator.add)
    def product(self): return self.reduce(operator.mul)

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

    def _new(self, items):
        return type(self)(items)

    def map(self, f, *args, **kwargs):
        g = (bind(f,*args,**kwargs) if callable(f)
             else f.format if isinstance(f,str)
             else f.__getitem__)
        return self._new(map(g, self))

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
