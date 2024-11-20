import torch
from abc import ABC, abstractmethod
import os
import keras
from keras.layers import *

os.environ["KERAS_BACKEND"] = "torch"
# torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')

class Model(ABC):
    @abstractmethod
    def forward(self, inputs):
        ...
    @abstractmethod
    def fit(self, data_generator, epochs=-1, callbacks=[], **kwargs):
        ...

class RecursiveNeuralNetworkFromScratch(Model):
	def __init__(self, hidden_size = 100, tokens=2, lr = 1e-1, var_init = .01):
		self.hidden_size = hidden_size
		self.lr = lr

		self.weights = dict(
			Wxh = torch.normal(0, var_init, (hidden_size, tokens), requires_grad=True), # input to hidden
			Whh = torch.normal(0, var_init, (hidden_size, hidden_size), requires_grad=True), # hidden to hidden
			Why = torch.normal(0, var_init, (tokens, hidden_size), requires_grad=True), # hidden to output
			bh = torch.normal(0, var_init, (hidden_size,), requires_grad=True), # hidden bias
			by = torch.normal(0, var_init, (tokens,), requires_grad=True), # output bias
		)

		self.memory = dict(
			mWxh = torch.zeros((hidden_size, tokens), requires_grad=True), # input to hidden
			mWhh = torch.zeros((hidden_size, hidden_size), requires_grad=True), # hidden to hidden
			mWhy = torch.zeros((tokens, hidden_size), requires_grad=True), # hidden to output
			mbh = torch.zeros((hidden_size,), requires_grad=True), # hidden bias
			mby = torch.zeros((tokens,), requires_grad=True), # output bias
		)


	def forward(self, inputs):
		Wxh, Whh, Why, bh, by = self.weights.values()
		h = torch.zeros((self.hidden_size,))
		result = []

		for t in range(len(inputs)):
			h = torch.tanh( Wxh @ inputs[t] +  Whh @ h + bh)
			p = torch.nn.functional.softmax(Why @ h + by, dim=0)
			result.append((p, h))

		P, H = zip(*result)
		return dict(distribution = torch.stack(P), latent = torch.stack(H))


	def fit(self, data_generator, epochs=-1, callbacks=[]):
		while (epochs := epochs - 1) + 1:
			inputs, targets = data_generator() # input_length x tokens
			P, H = self.forward(inputs).values()

			# loss = -torch.sum(P * torch.log(torch.sum(P * targets, dim=1))[:, None])
			loss = -torch.sum(torch.sum(P, dim=1) * torch.log(torch.sum(P * targets, dim=1)))
			loss.backward()

			for callback in callbacks:
				callback(loss = loss, distribution = P, latent = H, inputs = inputs, epoch=epochs)

			with torch.no_grad():
				for param, mem in zip(self.weights.values(), self.memory.values()):
					mem += param.grad.mul(param.grad)
					param -= self.lr * param.grad / torch.sqrt(mem + 1e-8)
					param.grad.zero_()
			
			yield loss.item()


class ConvolutionalClassifier(keras.Model):
	def __init__(self, input_length, labels=2, lr=1e-3, model_scale=4):
		inputs = keras.Input(shape=(input_length, 1))
		x = Conv1D(filters=int(4*model_scale), kernel_size=3, activation='relu')(inputs)
		x = BatchNormalization()(x)
		x = MaxPooling1D(pool_size=2)(x)

		x = Conv1D(filters=int(16*model_scale), kernel_size=3, activation='relu')(x)
		x = BatchNormalization()(x)
		x = MaxPooling1D(pool_size=2)(x)

		x = Conv1D(filters=int(32*model_scale), kernel_size=3, activation='relu')(x)
		x = BatchNormalization()(x)
		x = MaxPooling1D(pool_size=2)(x)

		x = Flatten()(x)
		x = Dense(int(64*model_scale), activation='relu')(x)
		x = Dropout(0.5)(x)
		x = Dense(labels, activation='linear')(x)

		super().__init__(inputs=inputs, outputs=x)
		self.lr = lr

	def fit(self, data_generator, epochs=-1, callbacks=[]):
		optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

		while (epochs := epochs - 1) + 1:
			inputs, targets = data_generator()		
			logits = self(inputs)
			loss = torch.nn.functional.cross_entropy(logits, targets)

			loss.backward()
			optimizer.step()
			self.zero_grad()

			for callback in callbacks:
				callback(loss = loss, inputs = inputs, targets=targets, logits=logits, epoch=epochs)


