import numpy as np
import torch.nn as nn
import torch.tensor as tensor
import matplotlib.pyplot as plt
import re, time, torch, math, types, functools

from functools import partial
from collections import defaultdict
from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time

from .data import L
from .utils import\
        camel2snake,\
        listify,\
        _is_instance,\
        _is_first,\
        sort_by_run

class Callback:
    run_before, run_after, toward_end = None, None, None
    def set_learner(self, run):
        self.run = run

    @property
    def name(self):
        name = re.sub(r"Callback$", "", self.__class__.__name__)
        return camel2snake(name or "callback")

    def __getattr__(self, attr):
        """Pass all attributes from learner to callback."""
        return getattr(self.run, attr)

    def __call__(self, cb_name):
        """Call callback, if return True signal a stop else continue."""
        f = getattr(self, cb_name, None)
        if f and f(): return True
        return False


class TrainEvalCallback(Callback):
    """Required callback that handles epoch and iterations values."""
    def begin_fit(self):
        """Clear out value for training."""
        self.run.n_epochs=0.
        self.run.n_iter=0

    def after_batch(self):
        """Aggreate iterations and epochs."""
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1

    def begin_epoch(self):
        self.run.n_epochs = self.epoch
        self.model.train()
        self.run.in_train=True

    def begin_validate(self):
        self.model.eval()
        self.run.in_train=False


class CancelTrainException(Exception): pass
class CancelEpochException(Exception): pass
class CancelBatchException(Exception): pass


class Metric:
    def __init__(self): self.reset()
    def reset(self):
        self.total, self.count = 0., 0

    def __call__(self, learn, **kwargs): pass

    @property
    def name(self): return "metric"
    @property
    def value(self): return self.total/self.count

class AvgLoss(Metric):
    def __call__(self, learn, **kwargs):
        bs = learn.xb.shape[0]
        self.count += bs
        self.total += learn.loss.detach().cpu()*bs 

    @property
    def name(self): return "loss"

class AvgSmoothedLoss(Metric):
    def __init__(self, beta=0.98, ignore_reset=True):
        self.beta = beta
        self.ignore_reset = ignore_reset
        self.total, self.count = torch.tensor(0.), 0

    def reset(self):
        if self.ignore_reset: return
        self.total, self.count = torch.tensor(0.), 0

    def __call__(self, learn, **kwargs):
        self.count += 1
        self.total = torch.lerp(learn.loss.detach().cpu(), self.total, self.beta)

    @property
    def name(self): return "smooth_loss"
    @property
    def value(self): return self.total/(1-self.beta**self.count)

class AvgMetric(Metric):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def __call__(self, learn, **kwargs):
        bs = run.xb.shape[0]
        self.count += bs
        self.total += self.func(learn.pred, learn.yb)*bs

    @property
    def name(self): return self.func.__name__


class Recorder(Callback):
    run_after = TrainEvalCallback
    def begin_fit(self):
        self.train_records = defaultdict(list)
        self.valid_records = defaultdict(list)
        self.train_records["lr"]
        self.epoch_iters = []

    def after_batch(self):
        if self.in_train:
            self.train_records["lr"].append(self.opt.hypers[-1]['lr'])
            if hasattr(self, "avg_stats") and self.in_train:
                for metric in self.avg_stats.train_metrics:
                    self.train_records[metric.name].append(metric.value)

    def after_epoch(self):
        if hasattr(self, "avg_stats") and not self.in_train:
            for metric in self.avg_stats.valid_metrics:
                self.valid_records[metric.name].append(metric.value)
            self.epoch_iters.append(self.n_iter)
                
    def plot_loss(self, skip_start=5, with_valid=True, log_xaxes=False):
        losses = self.train_records["smooth_loss"]
        plt.plot(np.linspace(skip_start, self.n_iter, len(losses)),
            losses, label="Train Smoothed")
        if with_valid:
            losses = self.valid_records["loss"]
            plt.plot(self.epoch_iters, losses, label="Valid")
        if log_xaxes: plt.xscale('log')
        plt.legend()

class AvgStatsCallback(Callback):
    run_before, run_after = Recorder, TrainEvalCallback
    def __init__(self, train_metrics=[], valid_metrics=[]):
        self.train_metrics = L(AvgLoss(), AvgSmoothedLoss()) + train_metrics
        self.valid_metrics = L(AvgLoss(), AvgSmoothedLoss()) + valid_metrics

    def begin_fit(self):
        names = L("epoch") +\
                self.train_metrics.attrgot("name").map("train_{}") +\
                self.valid_metrics.attrgot("name").map("valid_{}")
        self.logger(names)

    def begin_epoch(self):
        for metrics in [self.train_metrics, self.valid_metrics]:
            for metric in metrics: metric.reset()
        self.start_time = time.time()

    def after_loss(self):
        metrics = self.train_metrics if self.in_train else self.valid_metrics
        with torch.no_grad():
            for metric in metrics:
                metric(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_metrics, self.valid_metrics]:
            for m in o: 
                stats += [f"{m.value:.6f}"]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)

class ProgressCallback(Callback):
    _order=-1
    def begin_fit(self):
        self.mbar = master_bar(range(self.epochs))
        self.mbar.on_iter_begin()
        self.run.logger = partial(self.mbar.write, table=True)

    def after_fit(self):      self.mbar.on_iter_end()
    def after_batch(self):    self.pb.update(self.iter)
    def begin_epoch   (self): self.set_pb()
    def begin_validate(self): self.set_pb()

    def set_pb(self):
        #self.pb = progress_bar(self.data_loader, parent=self.mbar, auto_update=False)
        self.pb = progress_bar(self.data_loader, parent=self.mbar)
        self.mbar.update(self.epoch)


class CudaCallback(Callback):
    def begin_fit(self):
        """Place all model parameters to gpu."""
        if torch.cuda.is_available():
            self.model.cuda()

    def begin_batch(self):
        """Place all batch data to gpu."""
        if torch.cuda.is_available():
            self.run.xb, self.run.yb = self.xb.cuda(), self.yb.cuda()


class BatchTransformCallBack(Callback):
    """Callback to perform transformations on batch using a
    transformation function."""
    _order=2
    def __init__(self, trans_func):
        self.trans_func = trans_func

    def begin_batch(self): self.run.xb = self.trans_func(self.xb)


def view_trans(*size):
    """Reshape batch to shape size."""
    def _inner(x): return x.view(((-1,) + size))
    return _inner


class HookCallBack(Callback):
    _order = 1
    def __init__(self, hook_func):
        self.hook_func = hook_func

    def begin_fit(self):
        self.hooks = []
        self.hook_names = []

        for layer, param in enumerate(self.model):

            if isinstance(param, nn.Sequential):
                if isinstance(param[0], nn.Conv2d):
                    self.hook_names.append("Conv2d_" + str(layer))
                    self.hooks.append(Hook(param, self.hook_func))

            elif isinstance(param, nn.Linear):
                self.hook_names.append("Linear_" + str(layer))
                self.hooks.append(Hook(param, self.hook_func))


    def after_fit(self):
        for hook in self.hooks:
            hook.remove()

class Hook:
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()


def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, "stats"): hook.stats = ([], [])
    means, stds = hook.stats
    means.append(outp.data.mean().cpu())
    stds.append(outp.data.std().cpu())


def annealer(f):
    "Decorator to make `f` return itself partially applied."
    @functools.wraps(f)
    def _inner(start, end): return partial(f, start, end)
    return _inner

@annealer
def SchedLin(start, end, pos):
    "Linear schedule function from `start` to `end`"
    return start + pos*(end-start)

@annealer
def SchedCos(start, end, pos): 
    "Cosine schedule function from `start` to `end`"
    return start + (1 + math.cos(math.pi*(1-pos))) * (end-start)/2

@annealer
def SchedNo(start, end, pos):
    "Constant schedule function with `start` value"
    return start

@annealer
def SchedExp(start, end, pos):
    "Exponential schedule function from `start` to `end`"
    return start * (end/start)**pos

def SchedPoly(start, end, power):
    "Polynomial schedule (of `power`) function from `start` to `end`"
    def _inner(pos): return start + (end - start) * pos ** power
    return _inner

def combine_scheds(pcts, scheds):
    "Combine `scheds` according to `pcts` in one function"
    assert sum(pcts) == 1.
    pcts = tensor([0] + L(pcts))
    assert torch.all(pcts >= 0)
    pcts = torch.cumsum(pcts, 0)
    def _inner(pos):
        if pos == 1.: return scheds[-1](1.)
        idx = (pos >= pcts).nonzero().max()
        actual_pos = (pos-pcts[idx]) / (pcts[idx+1]-pcts[idx])
        return scheds[idx](actual_pos.item())
    return _inner

def combined_cos(pct, start, middle, end):
    "Return a scheduler with cosine annealing from `start`->`middle` & `middle` -> `end`."
    return combine_scheds([pct, 1-pct], [SchedCos(start, middle), SchedCos(middle, end)])


class ParamSchedulerCallback(Callback):
    "Class to dynamically modify hyper-parameters"
    run_after = TrainEvalCallback
    def __init__(self, sched_dict):
        "Input is key=hyper-value and value=sched."
        self.sched_funcs = sched_dict

    def begin_fit(self):
        "Create record of dynamic hyper-parameters and set to learner."
        self.hypers = {hyp_nam:[] for hyp_nam in self.sched_funcs.keys()}

    def begin_batch(self):
        if not self.in_train: return
        for hyp_nam, func in self.sched_funcs.items():
            self.opt.set_hyper(hyp_nam, func(self.n_epochs/self.epochs))

    def after_batch(self):
        "Record hyper-parameter values."
        if not self.in_train: return
        for hyp_nam in self.sched_funcs.keys():
            self.hypers[hyp_nam].append(self.opt.hypers[-1][hyp_nam])

    def after_fit(self):
        if hasattr(self.run, 'recorder'):
            self.recorder.hypers = self.hypers


class LRFind(ParamSchedulerCallback):
    run_after=Recorder
    def __init__(self, max_iter=100, min_lr=1e-7, max_lr=10, factor=4, stop_div=True):
        self.factor = factor
        self.max_iter, self.stop_div = max_iter, stop_div
        self.sched_funcs = {'lr': SchedExp(min_lr, max_lr)}

    def begin_fit(self):
        super().begin_fit()
        self.best_loss = float('inf')

    def begin_batch(self):
        if not self.in_train: return
        for hyp_nam, func in self.sched_funcs.items():
            self.opt.set_hyper(hyp_nam, func(self.n_iter/self.max_iter))

    def after_batch(self):
        super().after_batch()
        loss = self.recorder.train_records["smooth_loss"][-1]
        if loss < self.best_loss: self.best_loss = loss
        if loss > self.factor*self.best_loss and self.stop_div: raise CancelTrainException()
        if self.n_iter >= self.max_iter: raise CancelTrainException()

    def plot_loss(self, train_loss=False, log_xaxes=True):
        lrs = self.hypers["lr"]
        fig, ax = plt.subplots(1,1)

        losses = self.recorder.train_records["smooth_loss"]
        ax.plot(lrs, losses, label="Train Loss Smoothed")
        ax.set_xlabel("Learning Rate"); ax.set_ylabel("Loss")
        if log_xaxes: ax.set_xscale('log')
        if train_loss:
            losses = self.recorder.train_records["loss"]
            ax.plot(lrs, losses, label="Train Loss")
        ax.legend()

    def after_cancel_train(self):
        pass
