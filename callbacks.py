from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from automatons import Automaton

class Callback:
	def __call__(self, **kwargs):
		...

@dataclass
class Projection:
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