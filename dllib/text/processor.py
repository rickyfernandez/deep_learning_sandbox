from ..processor import Processor
from ..utils import uniqueify

class TokenProcessor(Processor):
    """
    Process a list of text sentances to single character tokens.
    This is intended to be used with crnn network where we insert
    a special blank character.
    """ 
    def __init__(self, blank='-'):
        self.vocab = None
        self.blank = blank
        
    def __call__(self, items):
        # vocab is defined on first use
        if self.vocab is None:
            vocab = list(token for word in items for token in word.lower())
            self.vocab = uniqueify(vocab)
            self.vocab.insert(0, self.blank)
            self.otoi = {v:k for k,v in enumerate(self.vocab)}
        return [self.encode(words) for words in items]
    
    def encode(self, item):
        return [self.otoi[token] for token in item.lower()]
    
    def decode(self, idx):
        assert self.vocab is not None
        return [self.vocab[a] for a in idx]
    
    def decode_batch(self, idxs):
        assert self.vocab is not None
        return [self.decode(idx) for idx in idxs]
