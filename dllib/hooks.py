from functools import partial

class Hook:
    def __init__(self, m, f): self.hook = m.register_forward_hook(partial(f, self))
    def remove(self): self.hook.remove()
    def __del__(self): self.remove()

def append_stats(hook, mod, inp, outp):
    if not hasattr(hook, "stats"): hook.stats = ([], [])
    means, stds = hook.stats
    means.append(outp.data.mean().cpu())
    stds.append(outp.data.std().cpu())

#class Hooks(ListContainer):
#    def __init__(self, ms, f): super().__init__([Hook(m,f) for m in ms])
#    def __enter__(self, *args): return self
#    def __exit__(self, *args): self.remove()
#    def __del__(self): self.remove()
#
#    def __delitem__(self, i):
#        self[i].remove()
#        super().__delitem__(i)
#
#    def remove(self):
#        for h in self: h.remove()
