from torch import nn
import re, time, torch
from functools import partial
import matplotlib.pyplot as plt

from fastprogress import master_bar, progress_bar
from fastprogress.fastprogress import format_time
from .utils import camel2snake, listify


class Callback:
    _order=0

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


class AvgStats:
    """
    Class to hold Statistics of list of metrics. We hold total cumulative statistics
    and counts separatley to have consits metrics. Metrics should be defined per
    batch and this class will multiply out batch count. Values are only kept per
    mini-batch and the discarded for next batch."""
    def __init__(self, metrics, in_train):
        self.metrics, self.in_train = listify(metrics), in_train

    def reset(self):
        self.tot_loss, self.count = 0.,0
        self.tot_mets = [0.]*len(self.metrics)

    @property
    def all_stats(self): return [self.tot_loss.item()] + self.tot_mets

    @property
    def avg_stats(self): return [o/self.count for o in self.all_stats]

    def __repr__(self):
        if not self.count: return ""
        return f"{'train' if self.in_train else 'valid'}: {self.avg_stats}"

    def accumulate(self, run):
        bn = run.xb.shape[0]
        self.tot_loss += run.loss * bn
        self.count += bn
        for i,m in enumerate(self.metrics):
            self.tot_mets[i] += m(run.pred, run.yb) * bn


class AvgStatsCallback(Callback):
    def __init__(self, metrics):
        self.train_stats = AvgStats(metrics, True)
        self.valid_stats = AvgStats(metrics, False)

    def begin_fit(self):
        met_names = ["loss"] + [m.__name__ for m in self.train_stats.metrics]
        names = ["epoch"] + [f"train_{n}" for n in met_names] + [
            f"valid_{n}" for n in met_names] + ["time"]
        self.logger(names)

    def begin_epoch(self):
        self.train_stats.reset()
        self.valid_stats.reset()
        self.start_time = time.time()

    def after_loss(self):
        stats = self.train_stats if self.in_train else self.valid_stats
        with torch.no_grad(): stats.accumulate(self.run)

    def after_epoch(self):
        stats = [str(self.epoch)]
        for o in [self.train_stats, self.valid_stats]:
            stats += [f"{v:.6f}" for v in o.avg_stats]
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
        self.pb = progress_bar(self.data_loader, parent=self.mbar, auto_update=False)
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


#class ParamScheduler(Callback):
#    _order = 1s
#    def __init__(self, pname, sched_funcs):
#        self.pname, self.sched_funcs, = pname, sched_funcs
#
#    def begin_fit(self):
#        if not isinstance(self.sched_funcs, (list, tuple)):
#            self.sched_funcs = [self.sched_funcs] * len(self.opt.param_groups)
#
#    def set_param(self):
#        assert len(self.opt.param_groups) == len(self.sched_funcs)
#        for pg, f in zip(self.opt.param_groups, self.sched_funcs):
#            pg[self.pname] = f(self.n_epochs/self.epochs)
#
#    def begin_batch(self):
#        if self.in_train: self.set_param()
#
#def annealer(f):
#    def _inner(start, end):
#        return partial(f, start, end)
#    return _inner
#
#@annealer
#def sched_lin(start, end, pos):
#    return start + pos*(end-start)
#
#@annealer
#def sched_cos(start, end, pos):
#    return start + (1 + math.cos(math.pi*(1-pos)))*(end-start)/2
#
#@annealer
#def sched_no(start, end, pos):
#    return start
#
#@annealer
#def sched_exp(start, end, pos):
#    return start * (end/start) ** pos
#
#def combine_scheds(pcts, scheds):
#    assert sum(pcts) == 1.
#    pcts = tensor([0] + listify(pcts))
#    assert torch.all(pcts >= 0)
#    pcts = torch.cumsum(pcts, 0)
#    def _inner(pos):
#        idx = (pos >= pcts).nonzero().max()
#        actual_pos = (pos-pcts[idx])/(pcts[idx+1]-pcts[idx])
#        return scheds[idx](actual_pos)
#    return _inner
#
#def cos_1cycle_anneal(start, high, end):
#    return [sched_cos(start, high), sched_cos(high, end)]
#
class Recorder(Callback):
    def begin_fit(self): self.lrs, self.losses = [], []

    def after_batch(self):
        if not self.in_train: return
        #self.lrs.append(self.opt.hypers[-1]['lr'])
        self.losses.append(self.loss.detach().cpu())

    def plot_lr  (self): plt.plot(self.lrs)
    def plot_loss(self): plt.plot(self.losses)

    def plot(self, skip_last=0):
        losses = [o.item() for o in self.losses]
        n = len(losses)-skip_last
        plt.xscale('log')
        plt.plot(self.lrs[:n], losses[:n])

#class ParamScheduler(Callback):
#    _order=1
#    def __init__(self, pname, sched_funcs):
#        self.pname, self.sched_funcs = pname, listify(sched_funcs)
#
#    def begin_batch(self):
#        if not self.in_train: return
#        fs = self.sched_funcs
#        if len(fs)==1: fs = fs*len(self.opt.param_groups)
#        pos = self.n_epochs/self.epochs
#        for f, h in zip(fs,self.opt.hypers): h[self.pname] = f(pos)
#
#class LR_Find(Callback):
#    _order=1
#    def __init__(self, max_iter=100, min_lr=1e-6, max_lr=10):
#        self.max_iter,self.min_lr,self.max_lr = max_iter,min_lr,max_lr
#        self.best_loss = 1e9
#
#    def begin_batch(self):
#        if not self.in_train: return
#        pos = self.n_iter/self.max_iter
#        lr = self.min_lr * (self.max_lr/self.min_lr) ** pos
#        for pg in self.opt.hypers: pg['lr'] = lr
#
#    def after_step(self):
#        if self.n_iter>=self.max_iter or self.loss>self.best_loss*10:
#            raise CancelTrainException()
#        if self.loss < self.best_loss: self.best_loss = self.loss
