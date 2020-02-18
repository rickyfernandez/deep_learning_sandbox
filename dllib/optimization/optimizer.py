from functools import partial
from ..utils import listify, compose


class Optimizer:
    """Base class for all optimizers."""
    def __init__(self, params, steppers, **defaults):
        """
        Optimizer for updating gradients, params are the variables to
        update with steppers which is a list of functions to modify
        the parameters with/without gradients.
        """
        self.steppers = listify(steppers)
        maybe_update(self.steppers, defaults, get_defaults)

        # might be a generator
        self.param_groups = list(params)
        if not isinstance(self.param_groups[0], list):
            self.param_groups = [self.param_groups]
        self.hypers = [{**defaults} for p in self.param_groups]

    def grad_params(self):
        """Return list of parameters (must have gradients) with hyperparameter values."""
        return [(p,hyper) for pg,hyper in zip(self.param_groups, self.hypers)
            for p in pg if p.grad is not None]

    def zero_grad(self):
        """Zero out gradients of all parameters stored."""
        for param, hyper in self.grad_params():
            param.grad.detach_()
            param.grad.zero_()

    def step(self):
        """Step through all updating functions for all parameters."""
        for p, hyper in self.grad_params(): compose(p, self.steppers, **hyper)

    def __repr__(self):
        string = ""
        for pg in self.param_groups:
            for p in pg:
                string += f"{p.shape}\n"
        return string

def sgd_step(p, lr, **kwargs):
    """Updatae weights from gradients and learning rate"""
    # multiply then add
    p.data.add_(-lr, p.grad.data)
    return p


def weight_decay(p, lr, wd, **kwargs):
    """Penalize weights with weight decay"""
    p.data.mul_(1 - lr*wd)
    return p
weight_decay._defaults = dict(wd=0.)


def l2_reg(p, lr, wd, **kwargs):
    p.grad.data.add_(wd, p.data)
    return p
l2_reg._defaults = dict(wd=0.)


SgdOptimizer = partial(Optimizer, steppers=[weight_decay, sgd_step])


def maybe_update(os, dest, f):
    for o in os:
        for k, v in f(o).items():
            if k not in dest: dest[k] = v

def get_defaults(d):
    return getattr(d, '_defaults', {})


#class StatefulOptimizer(Optimizer):
#    def __init__(self, params, steppers, stats=None, **defaults):
#        self.stats = listify(stats)
#        maybe_update(self.stats, defaults, get_defaults)
#        super().__init__(params, steppers, **defaults)
#        self.state = {}
#
#    def step(self):
#        for p,hyper in self.grad_params():
#            if p not in self.state:
#                # Create a state for p and call all the statistics
#
#                self.state[p] = {}
#                maybe_update(self.stats, self.state[p], lambda o: o.init_state(p))
#
#            state = self.state[p]
#            for stat in self.stats: state = stat.update(p, state, **hyper)
#            compose(p, self.steppers, **state, **hyper)
#            self.state[p] = state
#
#
#class Stat:
#    _defaults = {}
#    def ini_state(self, p): raise NotImplementedError
#    def update(self, p, state, **kwargs): raise NotImplementedError
#
#def momentum_step(p, lr, grad_avg, **kwargs):
#    p.data.add_(-lr, grad_avg)
#    return p
#
#def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2
#
#class AverageSGrad(Stat):
#    _defaults = dict(mom=0.9)
#
#    def __init__(self, dampening:bool=False): self.dampening=dampening
#    def init_state(self, p): return {'grad_avg': torch.zeros_like(p.grad.data)}
#    def update(self, p, state, mom, **kwargs):
#        state["mom_damp"] = 1-mom if self.dampening else 1.
#        state["grad_avg"].mul_(mom).add_(state["mom_damp"], p.grad.data)
#        return state
#
#class AverageSqrGrad(Stat):
#    _defaults = dict(sqr_mom=0.99)
#
#    def __init__(self, dampening:bool=True): self.dampening=dampening
#    def init_state(self, p): return {'sqr_avg': torch.zeros_like(p.grad.data)}
#    def update(self, p, state, sqr_mom, **kwargs):
#        state['sqr_damp'] = 1-sqr_mom if self.dampening else 1.
#        state['sqr_avg'].mul_(sqr_mom).addcmul_(state['sqr_damp'], p.grad.data, p.grad.data)
#        return state
#
#class StepCount(Stat):
#    def init_state(self, p): return {"step": 0}
#    def update(self, p, state, **kwargs):
#        state['step'] += 1
#        return state
#
#def debias(mom, damp, step): return damp * (1-mom**step)/(1-mom)
#
#def adam_step(p, lr, mom, mom_damp, step, sqr_mom, sqr_damp, grad_avg, sqr_avg, eps, **kwargs):
#    debias1 = debias(mom,     mom_damp, step)
#    debias2 = debias(sqr_mom, sqr_damp, step)
#    p.data.addcdiv_(-lr / debias1, grad_avg, (sqr_avg/debias2).sqrt() + eps)
#    return p
#adam_step._defaults = dict(eps=1e-5)
#
#def adam_opt(xtra_step=None, **kwargs):
#    return partial(StatefulOptimizer, steppers=[adam_step,weight_decay]+listify(xtra_step),
#                   stats=[AverageGrad(dampening=True), AverageSqrGrad(), StepCount()], **kwargs)
