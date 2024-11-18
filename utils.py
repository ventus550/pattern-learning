import torch
from typing import Callable, Iterable
from automatons import Automaton

def make_tokenizer(mapdict):
	tokenizer = {'A': 0, 'B': 1, '#': 2}
	tokenizer.update({tokenizer[k]: k for k in tokenizer})
	return tokenizer

def onehot(sequence: Iterable, tokens: int):
	return torch.nn.functional.one_hot(torch.tensor(sequence), num_classes=tokens).float()

def data_loader(automaton: Automaton):
	while True:
		sample = torch.tensor([*list(automaton.generator), automaton.tokens])
		sample = onehot(sample, automaton.tokens+1)
		inputs  = sample[:-1]
		targets = sample[1:]
		yield inputs, targets