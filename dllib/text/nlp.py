import torch
from torch import tensor
import re, collections
from pathlib import Path
from typing import Collection
from collections import Counter

import spacy, html
from spacy.symbols import ORTH
from concurrent.futures import ProcessPoolExecutor
from fastprogress import progress_bar

from ..datablock.data import\
        ItemList,\
        Processor,\
        DataBunch,\
        DataLoader

from ..utils import\
        get_files,\
        compose


def read_file(fn):
    with open(fn, 'r', encoding='utf8') as f: return f.read()

class TextList(ItemList):
    """
    List that holds file names for each text samlple.
    """
    @classmethod
    def from_files(cls, path, extensions='.txt', recurse=True, include=None, **kwargs):
        return cls(get_files(path, extensions, recurse=recurse, include=include), path, **kwargs)

    def get(self, i):
        if isinstance(i, Path): return read_file(i)
        return i


UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ = "xxunk xxpad xxbos xxeos xxrep xxwrep xxup xxmaj".split()


def sub_br(t):
    "Replaces the <br /> by \n"
    re_br = re.compile(r'<\s*br\s*/?>', re.IGNORECASE)
    return re_br.sub("\n", t)


def spec_add_spaces(t):
    "Add spaces around / and #"
    return re.sub(r'([/#])', r' \1 ', t)


def rm_useless_spaces(t):
    "Remove multiple spaces"
    return re.sub(' {2,}', ' ', t)


def replace_rep(t):
    "Replace repetitions at the character level: cccc -> TK_REP 4 c"
    def _replace_rep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_REP} {len(cc)+1} {c} '
    re_rep = re.compile(r'(\S)(\1{3,})')
    return re_rep.sub(_replace_rep, t)


def replace_wrep(t):
    "Replace word repetitions: word word word -> TK_WREP 3 word"
    def _replace_wrep(m:Collection[str]) -> str:
        c,cc = m.groups()
        return f' {TK_WREP} {len(cc.split())+1} {c} '
    re_wrep = re.compile(r'(\b\w+\W+)(\1{3,})')
    return re_wrep.sub(_replace_wrep, t)


def fixup_text(x):
    "Various messy things we've seen in documents"
    re1 = re.compile(r'  +')
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>',UNK).replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))


default_pre_rules = [fixup_text, replace_rep, replace_wrep, spec_add_spaces, rm_useless_spaces, sub_br]
default_spec_tok = [UNK, PAD, BOS, EOS, TK_REP, TK_WREP, TK_UP, TK_MAJ]


def replace_all_caps(x):
    "Replace tokens in ALL CAPS by their lower version and add `TK_UP` before."
    res = []
    for t in x:
        if t.isupper() and len(t) > 1: res.append(TK_UP); res.append(t.lower())
        else: res.append(t)
    return res


def deal_caps(x):
    "Replace all Capitalized tokens in by their lower version and add `TK_MAJ` before."
    res = []
    for t in x:
        if t == '': continue
        if t[0].isupper() and len(t) > 1 and t[1:].islower(): res.append(TK_MAJ)
        res.append(t.lower())
    return res


def add_eos_bos(x): return [BOS] + x + [EOS]


default_post_rules = [deal_caps, replace_all_caps, add_eos_bos]


def parallel(func, arr, max_workers=4):
    """Wrapper that applies the function func to items in list in parallel.
    Values in list are enumerated."""
    if max_workers<2: results = list(progress_bar(map(func, enumerate(arr)), total=len(arr)))
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            return list(progress_bar(ex.map(func, enumerate(arr)), total=len(arr)))
    if any([o is not None for o in results]): return results


class TokenizeProcessor(Processor):
    """Replaces words in a corpus by speical tokens and takes care of
    special language rules (i.e. don't). Can be run in parallel."""
    def __init__(self, lang="en", chunksize=2000, pre_rules=None, post_rules=None, max_workers=4): 
        self.chunksize = chunksize
        self.max_workers = max_workers
        self.tokenizer = spacy.blank(lang).tokenizer

        # add our special words into list of tokens
        for w in default_spec_tok:
            self.tokenizer.add_special_case(w, [{ORTH: w}])

        self.pre_rules  = default_pre_rules  if pre_rules  is None else pre_rules
        self.post_rules = default_post_rules if post_rules is None else post_rules

    def proc_chunk(self, args):
        """Process text first by pre_rules then by spacy and lastly post_rules.
        """
        i, chunk = args
        chunk = [compose(t, self.pre_rules) for t in chunk]
        docs  = [[d.text for d in doc] for doc in self.tokenizer.pipe(chunk)]
        docs  = [compose(t, self.post_rules) for t in docs]
        return docs

    def __call__(self, items):
        """Transform all values in data list into numerical values. Runs in
        parallel by batching items.
        """
        tokens = []
        if isinstance(items[0], Path): items = [read_file(i) for i in items]

        chunks = [items[i: i+self.chunksize] for i in range(0, len(items), self.chunksize)]
        tokens = parallel(self.proc_chunk, chunks, max_workers=self.max_workers)

        # concatentate all results from parallel workers
        return sum(tokens, [])

    def proc1(self, item): return self.proc_chunk((1,[item]))[0]

    def deprocess(self, tokens): return [self.deproc1(tok) for tok in tokens]
    def deproc1(self, tokens): return " ".join(tokens)


class NumericalizeProcessor(Processor):
    """Create vocabulary for text creating mapping between integers
    and words."""
    def __init__(self, vocab=None, max_vocab=60000, min_freq=2):
        self.vocab = vocab
        self.max_vocab = max_vocab
        self.min_freq = min_freq

    def __call__(self, items):
        """Map words to integers, if vocab mapping does not exist then
        create it."""
        if self.vocab is None:
            freq = Counter(word for sentance in items for word in sentance)
            self.vocab = [word for word,count in freq.most_common(self.max_vocab)
                    if count >= self.min_freq]

            # place our special tokens in front
            for word in reversed(default_spec_tok):
                if word in self.vocab: self.vocab.remove(word)
                self.vocab.insert(0, word)

        if getattr(self, 'otoi', None) is None:
            self.otoi = collections.defaultdict(int,{v:k for k,v in enumerate(self.vocab)})
        return [self.proc1(o) for o in items]

    def proc1(self, item):  return [self.otoi[o] for o in item]

    def deprocess(self, idxs):
        assert self.vocab is not None
        return [self.deproc1(idx) for idx in idxs]
    def deproc1(self, idx): return [self.vocab[i] for i in idx]


class LM_PreLoader():
    """Disregards the target and creates a new target from input shifted
    by unit. This is to only be used in language models predicting the
    next word. All items are concatenated to create a corpus then batched.
    The first line in the batch continues with the second first line in
    the second batch thus the hidden state can be carried over in rnn
    models.
    """
    def __init__(self, data, bs=64, bptt=70, shuffle=False):
        self.data,self.bs,self.bptt,self.shuffle = data,bs,bptt,shuffle
        total_len = sum([len(t) for t in data.x])
        self.n_batch = total_len // bs
        self.batchify()

    def __len__(self): return ((self.n_batch-1) // self.bptt) * self.bs

    def __getitem__(self, idx):
        source = self.batched_data[idx % self.bs]
        seq_idx = (idx // self.bs) * self.bptt
        return source[seq_idx:seq_idx+self.bptt],source[seq_idx+1:seq_idx+self.bptt+1]

    def batchify(self):
        # place all docs as single stream
        texts = self.data.x
        if self.shuffle: texts = texts[torch.randperm(len(texts))]
        stream = torch.cat([tensor(t) for t in texts])
        self.batched_data = stream[:self.n_batch * self.bs].view(self.bs, self.n_batch)


def get_lm_dls(train_ds, valid_ds, bs, bptt, **kwargs):
    return (DataLoader(LM_PreLoader(train_ds, bs, bptt, shuffle=True), batch_size=bs, **kwargs),
            DataLoader(LM_PreLoader(valid_ds, bs, bptt, shuffle=False), batch_size=2*bs, **kwargs))


def lm_databunchify(sd, bs, bptt, **kwargs):
    return DataBunch(*get_lm_dls(sd.train, sd.valid, bs, bptt, **kwargs))


def get_clas_dls(train_ds, valid_ds, bs, **kwargs):
    train_sampler = SortishSampler(train_ds.x, key=lambda t: len(train_ds.x[t]), bs=bs)
    valid_sampler = SortSampler(valid_ds.x, key=lambda t: len(valid_ds.x[t]))
    return (DataLoader(train_ds, batch_size=bs, sampler=train_sampler, collate_fn=pad_collate, **kwargs),
            DataLoader(valid_ds, batch_size=bs*2, sampler=valid_sampler, collate_fn=pad_collate, **kwargs))


def clas_databunchify(sd, bs, **kwargs):
    return DataBunch(*get_clas_dls(sd.train, sd.valid, bs, **kwargs))
