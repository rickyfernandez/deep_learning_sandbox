import torch
from functools import partial
from collections import defaultdict

from ..data import L
from ..utils import listify, compose, merge, range_of, even_mults


def _update(state, new=None):
    if new is None: return state
    if isinstance(new, dict):
        state.update(new)
    return state

class BaseOptimizer:
    """Base functionality of all optimizers. Pytorch parameters are referenced
    by this class. Each parameter keeps a reference to its parameter group,
    state, and hyper-paramters. A state is the current value of a parameter
    that a callback update. During initialization a master default dict is
    created by extracting all parameters from the callbacks."""
    _keep_on_clear = ['force_train', 'do_wd']

    def __init__(self, params, cbs, train_bn=True, **defaults):
        """During initialization a master default dict is created by
        extracting all parameters from the callbacks and are assigned
        to each parameter.

        params: pytorch parameters
        cbs: callbacks to perform on each parameter
        """
        params = L(params)
        self.cbs = L(cbs)
        self.train_bn = train_bn
        self.state = defaultdict(dict)

        # merge all default values into one master default
        defaults = merge(*self.cbs.attrgot('defaults'), defaults)
        self.param_groups = L(L(p) for p in params)\
                if isinstance(params[0], (L, list)) else L([params])

        self.hypers = L({} for _ in range_of(self.param_groups))
        self.set_hypers(**defaults)
        self.frozen_idx = 0

    def all_params(self, n=slice(None), with_grad=False):
        """Return list parameters, parameter group, state and hyper-parameter
        value. Can be all the parameters or up to `n`.

        n: slice or integer
        with_grad: include parameters only with gradients"""
        res = L((p, pg, self.state[p], hyper)\
                for pg,hyper in zip(self.param_groups[n], self.hypers[n])\
                for p in pg)
        return L(o for o in res if o[0].grad is not None) if with_grad else res

    def set_hypers(self, **kwargs):
        """Set hyperparameters for all parameters by using a dictionary of
        hyper-values (i.e. {'lr': 0.1, 'mom': 0.8}).

        kwargs: dictionary of hyperparameter values (hypername, val)
        """
        L(kwargs.items()).starmap(self.set_hyper)

    def _set_hyper(self, hyp_nam, hyp_val):
        """Set all hyper-parameters to hyper-value. Hyper-value must have
        the same length of parameters."""
        for _hyp_val,hyp_dict in zip(hyp_val, self.hypers):
            hyp_dict[hyp_nam] = _hyp_val

    def set_hyper(self, hyp_nam, hyp_val):
        """Set hyper-parameter across all hyper-parameters. Hyper-value
        can be a number or a slice. In case of slice it will be extrapolated
        across all parameters. If it is a slice with only end, then last
        parameter gets that value with the rest reduced by 10.
        
        hyp_nam: name of hyper-parameter
        hyp_val: value of hyper-parameter, can be number of slice"""
        if isinstance(hyp_val, slice):
            if hyp_val.start:
                hyp_val = even_mults(
                        hyp_val.start,
                        hyp_val.stop, len(self.param_groups))
            else:
                hyp_val = [hyp_val.stop/10]*(len(self.param_groups)-1) + [hyp_val.stop]

        hyp_val = L(hyp_val, use_list=None)
        if len(hyp_val)==1: hyp_val = hyp_val*len(self.param_groups)
        assert len(hyp_val) == len(self.hypers)
        self._set_hyper(hyp_nam, hyp_val)

    def zero_grad(self):
        "Zero all grad of parameters"
        for p,*_ in self.all_params(with_grad=True):
            p.grad.detach_()
            p.grad.zero_()

    def step(self):
        """Update state and execute steppers on all parameters that have grad.
        Parameters are updated inplace in callbacks and callback may return
        updated parameters."""
        for p,pg,state,hyper in self.all_params(with_grad=True):
            for cb in self.cbs: state = _update(state, cb(p, **{**state, **hyper}))
            self.state[p] = state

    def _set_require_grad(self, rg, p, pg, state, h):
        p.requires_grad_(rg or state.get('force_train', False))

    def freeze_to(self, n):
        "Freeze parameters groups up to `n`."
        self.frozen_idx = n if n >= 0 else len(self.param_groups) + n
        if self.frozen_idx >= len(self.param_groups):
            import sys
            sys.exit(0)
        for o in self.all_params(slice(n, None)): self._set_require_grad(True,  *o)
        for o in self.all_params(slice(None, n)): self._set_require_grad(False, *o)

    def freeze(self):
        "Freeze up to last parameter group."
        assert(len(self.param_groups)>1)
        self.freeze_to(-1)

    def set_freeze(n, rg, ignore_force_train=False):
        "Freeze parameter group."
        for p in self.param_groups[n]:
            p.requires_grad_(rg or (state.get('force_train', False)\
                    and not ignore_force_train))

    def unfreeze(self):
        "Unfreeze the whole model."
        self.freeze_to(0)

    def clear_state(self):
        "Reset the state of the optimizer but ignore any values in _keep_on_clear"
        for p,pg,state,hyper in self.all_params():
            self.state[p] = {k: state[k] for k in self._keep_on_clear if k in state}

    def state_dict(self):
        "Return the state of the optimizer in a dictionary"
        state = [self.state[p] for p,*_ in self.all_params()]
        return {'state': state, 'hypers' : self.hypers}

    def load_state_dict(self, state):
        "Load the content of `state` dictionary"
        assert len(state["hypers"]) == len(self.param_groups)
        assert len(state["state"]) == sum([len(pg) for pg in self.param_groups])
        self.hypers = state['hypers']
        self.state = {p: s for p,s in zip(self.all_params().itemgot(0), state['state'])}


def sgd_step(p, lr, **kwargs):
    p.data.add_(-lr, p.grad.data)


def weight_decay(p, lr, wd, do_wd=True, **kwargs):
    "Weight decay as decaying `p` with `lr*wd`"
    if do_wd and wd != 0: p.data.mul_(1 - lr*wd)
weight_decay.defaults = dict(wd=0.)


def l2_reg(p, lr, wd, do_wd=True, **kwargs):
    "L2 regularization as adding `wd*p` to `p.grad`"
    if do_wd and wd != 0: p.grad.data.add_(wd, p.data)
l2_reg.defaults = dict(wd=0.)


def average_grad(p, mom, dampening=False, grad_avg=None, **kwargs):
    "Keeps track of the avg grads of `p` in `state` with `mom`."
    if grad_avg is None: grad_avg = torch.zeros_like(p.grad.data)
    damp = 1-mom if dampening else 1.
    grad_avg.mul_(mom).add_(damp, p.grad.data)
    return {'grad_avg': grad_avg}
average_grad.defaults = dict(mom=0.9)


def average_sqr_grad(p, sqr_mom, dampening=True, sqr_avg=None, **kwargs):
    if sqr_avg is None: sqr_avg = torch.zeros_like(p.grad.data)
    damp = 1-sqr_mom if dampening else 1.
    sqr_avg.mul_(sqr_mom).addcmul_(damp, p.grad.data, p.grad.data)
    return {'sqr_avg': sqr_avg}
average_sqr_grad.defaults = dict(sqr_mom=0.99)


def momentum_step(p, lr, grad_avg, **kwargs):
    "Step for SGD with momentum with `lr`"
    p.data.add_(-lr, grad_avg)


def SGD(params, lr, mom=0., wd=0., decouple_wd=True):
    "A `Optimizer` for SGD with `lr` and `mom` and `params`."
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    if mom != 0: cbs.append(average_grad)
    cbs.append(sgd_step if mom==0 else momentum_step)
    return BaseOptimizer(params, cbs, lr=lr, mom=mom, wd=wd)

def rms_prop_step(p, lr, sqr_avg, eps, grad_avg=None, **kwargs):
    "Step for SGD with momentum with `lr`"
    denom = sqr_avg.sqrt().add_(eps)
    p.data.addcdiv_(-lr, (grad_avg if grad_avg is not None else p.grad), denom)
rms_prop_step.defaults = dict(eps=1e-8)


def RMSProp(params, lr, sqr_mom=0.99, mom=0., wd=0., decouple_wd=True):
    "A `Optimizer` for RMSProp with `lr`, `sqr_mom`, `mom` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += ([average_sqr_grad] if mom == 0. else [average_grad, average_sqr_grad])
    cbs.append(rms_prop_step)
    return BaseOptimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, wd=wd)


def step_stat(p, step=0, **kwargs):
    "Register the number of steps done in `state` for `p`"
    step += 1
    return {'step' : step}


def debias(mom, damp, step): return damp * (1-mom**step) / (1-mom)


def adam_step(p, lr, mom, step, sqr_mom, grad_avg, sqr_avg, eps, **kwargs):
    "Step for Adam with `lr` on `p`"
    debias1 = debias(mom, 1-mom, step)
    debais2 = debias(sqr_mom, 1-sqr_mom, step)
    p.data.addcdiv_(-lr/debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
    return p
adam_step._defaults = dict(eps=1e-5)


def Adam(params, lr, mom=0.9, sqr_mom=0.99, eps=1.e-5, wd=0., decouple_wd=True):
    "A `Optimizer` for Adam with `lr`, `mom`, `sqr_mom`, `eps` and `params`"
    cbs = [weight_decay] if decouple_wd else [l2_reg]
    cbs += [partial(average_grad, dampening=True), average_sqr_grad, step_stat, adam_step]
    return Optimizer(params, cbs, lr=lr, mom=mom, sqr_mom=sqr_mom, eps=eps, wd=wd)
