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
        listify

def _is_instance(f, gs):
    "Check if f is instance or equal to any object in g"
    tst = [g if type(g) in [type, 'function'] else g.__class__ for g in gs]
    for g in tst:
        if isinstance(f, g) or f == g:
            return True

def _is_first(f, gs):
    "Check if f comes before all other objects in gs"
    for o in L(getattr(f, 'run_after', None)):
        if _is_instance(o, gs): return False
    for g in gs:
        if _is_instance(f, L(getattr(g, 'run_before', None))):
            return False
    # okay to be first
    return True

def sort_by_run(fs):
    end = L(fs).attrgot('toward_end')
    inp, res = L(fs)[~end] + L(fs)[end], L()
    while len(inp):
        for i,o in enumerate(inp):
            if _is_first(o, inp):
                res.append(inp.pop(i))
                break
        else: raise Exception("Impossible to sort")
    return res

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

    def begin_epoch(self):
        self.run.n_epochs = self.epoch

    def before_train(self):
        self.model.train()
        self.run.in_train=True

    def before_validate(self):
        self.model.eval()
        self.run.in_train=False

    def after_batch(self):
        """Aggreate iterations and epochs."""
        if not self.in_train: return
        self.run.n_epochs += 1./self.iters
        self.run.n_iter += 1


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

class AvgStatsCallback(Callback):
    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_metrics, self.valid_metrics]:
            for m in o:
                stats += [f"{m.value:.6f}"]
        stats += [format_time(time.time() - self.start_time)]
        self.logger(stats)


class Recorder(Callback):
    run_after = TrainEvalCallback

    def __init__(self, add_time=True, train_metrics=False, valid_metrics=True, beta=0.98):
        self.add_time = add_time
        self.train_metrics = train_metrics
        self.valid_metrics = valid_metrics

        self.valid_loss = AvgLoss()
        self.train_smooth_loss = AvgSmoothedLoss(beta=beta)

        self.start_time = None
        self.epoch_iters, self.log= None, None
        self.names, self.metric_names = None, None
        self.train_losses, self.valid_losses = None, None

    def begin_fit(self):
        self.epoch_iters = []
        self.lrs, self.values = [],[]
        self.train_losses, self.valid_losses = [],[]

        names = self.metrics.attrgot('name')
        if self.train_metrics and self.valid_metrics:
            names = L('loss') + names
            names = names.map('train_{}') + names.map('valid_{}')
        elif self.valid_metrics: names = L('train_loss', 'valid_loss') + names
        else: names = L('train_loss') + names
        if self.add_time: names.append('time')
        self.metric_names = 'epoch' + names
        self.train_smooth_loss.reset()
        self.logger(self.metric_names)

    def begin_epoch(self):
        self.start_time = time.time()
        self.log = L(self.epoch)

    def before_train(self):
        for metric in self._train_metrics: metric.reset()

    def after_train(self):
        self.log += self._train_metrics.attrgot('value')

    def before_validate(self):
        for metric in self._valid_metrics: metric.reset()

    def after_validate(self):
        self.log += self._valid_metrics.attrgot('value')
        self.valid_losses.append(self.valid_loss.value)
        self.epoch_iters.append(self.n_iter)

    def after_batch(self):
        metrics = self._train_metrics if self.in_train else self._valid_metrics
        for metric in metrics: metric(self.run)

        if not self.in_train: return
        self.lrs.append(self.opt.hypers[-1]['lr'])
        self.train_losses.append(self.train_smooth_loss.value)

    def after_epoch(self):
        self.run.final_record = self.log.copy()
        self.values.append(self.run.final_record)
        if self.add_time: self.log.append(time.time() - self.start_time)
        self.logger(self.log)

    @property
    def _train_metrics(self):
        return L(self.train_smooth_loss) + (self.metrics if self.train_metrics else L())

    @property
    def _valid_metrics(self):
        return L(self.valid_loss) + (self.metrics if self.valid_metrics else L())

    def plot_loss(self, skip_start=0, with_valid=True, log_xaxes=False):
        plt.plot(np.linspace(skip_start, self.n_iter, len(self.train_losses)),
            self.train_losses, label="Train Smoothed")
        if with_valid:
            plt.plot(self.epoch_iters, self.valid_losses, label="Valid")
        if log_xaxes: plt.xscale('log')
        plt.legend()

class ProgressCallback(Callback):
    run_after = TrainEvalCallback
    run_before = Recorder
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
        loss = self.recorder.train_losses[-1]
        #loss = self.recorder.train_records["smooth_loss"][-1]
        if loss < self.best_loss: self.best_loss = loss
        if loss > self.factor*self.best_loss and self.stop_div: raise CancelTrainException()
        if self.n_iter >= self.max_iter: raise CancelTrainException()

    def plot_loss(self, train_loss=False, log_xaxes=True):
        lrs = self.hypers["lr"]
        fig, ax = plt.subplots(1,1)

        #losses = self.recorder.train_records["smooth_loss"]
        ax.plot(lrs, self.recorder.train_losses, label="Train Loss Smoothed")
        ax.set_xlabel("Learning Rate"); ax.set_ylabel("Loss")
        if log_xaxes: ax.set_xscale('log')
        ax.legend()

    def after_cancel_train(self):
        pass
