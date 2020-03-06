import time
import torch
from torch import tensor
from functools import partial

from .data import L
from .utils import listify
from .optimization.optimizer import SGD
from .callbacks import TrainEvalCallback, CancelTrainException, CancelEpochException, CancelBatchException, sort_by_run

def param_getter(m):
    return [p for p in m.parameters() if p.requires_grad]

class Learner:
    """
    Class that holds all components to train a model.
    """
    def __init__(self, model, data, loss_func, opt_func=SGD,
            lr=1e-2, splitter=param_getter, cbs=None, cb_funcs=None, metrics=None):

        self.model, self.data, self.loss_func = model, data, loss_func
        self.opt_func, self.lr, self.splitter = opt_func, lr, splitter
        self.in_train, self.logger, self.opt = False, print, None

        self.i_mb = 0
        self.epoch, self.data_loader = None, None
        self.iter, self.xb, self.yb = None, None, None
        self.pred, self.loss, self.train = None, None, None

        self.cbs = []
        self.add_cb(TrainEvalCallback())
        self.add_cbs(cbs)
        self.add_cbs(cbf() for cbf in listify(cb_funcs))

        self.metrics = L(metrics)

    def add_cbs(self, cbs):
        """Add list of callbacks as an attribute"""
        for cb in listify(cbs): self.add_cb(cb)

    def add_cb(self, cb):
        """Add callback as an attribute with reference to learner"""
        cb.set_learner(self)
        setattr(self, cb.name, cb)
        self.cbs.append(cb)

    def remove_cbs(self, cbs):
        """Remove callback"""
        for cb in listify(cbs): self.cbs.remove(cb)

    def one_batch(self, i, xb, yb):
        """Run through one batch in model for training or prediction"""
        try:

            self.iter = i
            self.xb, self.yb = xb, yb;                      self("begin_batch")
            self.pred = self.model(self.xb);                self("after_pred")
            self.loss = self.loss_func(self.pred, self.yb); self("after_loss")

            if not self.in_train: return

            self.loss.backward();                           self("after_backward")
            self.opt.step();                                self("after_step")
            self.opt.zero_grad()

        except CancelBatchException:                        self("after_cancel_batch")
        finally:                                            self("after_batch")

    def all_batches(self):
        """Run through all batches in data set"""
        self.iters = len(self.data_loader)
        try:
            for i, (xb,yb) in enumerate(self.data_loader):
                self.one_batch(i, xb, yb)

            if self.in_train: self.i_mb += 1
        except CancelEpochException: self("after_cancel_epoch")

    def do_begin_fit(self, epochs):
        """First call before fit is called"""
        self.epochs, self.loss = epochs, tensor(0.)
        self("begin_fit")

    def do_begin_epoch(self, epoch):
        """First call before begin epoch called"""
        self.epoch, self.data_loader = epoch, self.data.train_dl
        return self("begin_epoch")

    def fit(self, epochs, cbs=None, reset_opt=False):
        """Fit model on data"""
        self.add_cbs(cbs)
        if reset_opt or not self.opt:
            self.opt = self.opt_func(self.splitter(self.model), lr=self.lr)

        try:
            self.do_begin_fit(epochs)
            for epoch in range(epochs):
                self.do_begin_epoch(epoch); self('begin_epoch')
                self("before_train"); self.all_batches(); self("after_train")

                with torch.no_grad():
                    self.data_loader = self.data.valid_dl
                    self('before_validate'); self.all_batches(); self("after_validate")
                self("after_epoch")

        except CancelTrainException: self("after_cancel_train")
        finally:
            self("after_fit")
            self.remove_cbs(cbs)

    ALL_CBS = {"begin_batch", "after_pred", "after_loss", "after_backward", "after_step",
        "after_cancel_batch", "after_batch", "after_cancel_epoch", "begin_fit",
        "begin_epoch", "before_train", "after_train", "before_validate", "after_validate", "after_epoch",
        "after_cancel_train", "after_fit"}

    def __call__(self, cb_name):
        """Call every callback registered"""
        res = False
        assert cb_name in self.ALL_CBS
        for cb in sort_by_run(self.cbs):
            res = cb(cb_name) and res
        return res
