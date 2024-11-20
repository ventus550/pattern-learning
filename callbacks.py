from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Callable
import torch
import numpy as np
from abc import ABC, abstractmethod
from automatons import Automaton
from models import Model

class Callback(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        ...

@dataclass
class Projection(Callback):
	automaton: Automaton
	k_recent: int = 1000
	automaton_states = []
	latent_states = []

	def __call__(self, latent, inputs, epoch, method: PCA | TSNE = PCA, **kwargs):
		self.automaton.reset()
		self.automaton_states.extend(self.automaton.next(token.argmax()).state for token in inputs)
		self.automaton.reset()
		self.latent_states.extend(latent)


		if epoch % 1000 == 100:
			recent_automaton_states = self.automaton_states[-self.k_recent:]
			recent_latent_states = self.latent_states[-self.k_recent:]
			
			projected_latent_states = method(n_components=2).fit_transform(
				np.array([h.detach().numpy() for h in recent_latent_states])
			)

			fig, ax = plt.subplots()
			normalized_states = (
				(np.array(recent_automaton_states) - min(recent_automaton_states)) /
				(max(recent_automaton_states) - min(recent_automaton_states))
			)
			colors = plt.cm.viridis(normalized_states)

			for i in range(len(projected_latent_states) - 1):
				ax.plot(
					projected_latent_states[i:i+2, 0],  # x-coordinates of the two points
					projected_latent_states[i:i+2, 1],  # y-coordinates of the two points
					alpha=0.5,
					color=colors[i],
            		linewidth=0.5,
				)
			scatter = ax.scatter(*projected_latent_states.T, c=recent_automaton_states, zorder=2)

			legend1 = ax.legend(handles=scatter.legend_elements()[0], labels=range(len(self.automaton.transitions)+1), title="STATES")
			plt.show()

@dataclass
class RecursiveNeuralNetworkFromScratchSampleOutput:
	model: Model
	pattern: list
	eos: int
	init: Callable

	def __call__(self, epoch, **kwargs):
		Wxh, Whh, Why, bh, by = self.model.weights.values()
		x = self.init()
		ixes = []
		h = torch.zeros([self.model.hidden_size,])

		while True:
			h = torch.tanh(Wxh @ x +  Whh @ h + bh)
			p = torch.nn.functional.softmax(Why @ h + by, dim=0)
			ix = np.random.choice(range(self.eos+1), p=p.detach().numpy().ravel())
			x = torch.zeros((self.eos+1,))
			x[ix] = 1
			ixes.append(ix)
			if ix == self.eos:
				break
		print(f"{epoch: <16} {ixes[-len(self.pattern)-1:]}{ixes}")

@dataclass
class LogLoss:
	frequency: int = 1
	def __call__(self, loss, epoch, **kwargs):
		if not epoch % self.frequency:
			print(f"{epoch: <16}\t{loss}")

@dataclass
class LogAccuracy:
	frequency: int = 1
	def __call__(self, logits, targets, epoch, **kwargs):
		if not epoch % self.frequency:
			correct = (logits.argmax(dim=1) == targets.argmax(dim=1)).sum()
			accuracy = correct / len(targets)
			print(f"{epoch: <16}accuracy:\t{accuracy}")
