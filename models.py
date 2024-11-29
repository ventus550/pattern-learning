import torch
import os
import keras
import numpy
from keras.layers import (
    Embedding,
    LSTM,
    Flatten,
    Dense,
    BatchNormalization,
    MaxPooling1D,
    Conv1D,
    Dropout,
    Bidirectional,
    Attention,
)

os.environ["KERAS_BACKEND"] = "torch"
# torch.set_default_device('cuda' if torch.cuda.is_available() else 'cpu')


class RecursiveNeuralNetworkFromScratch:
    def __init__(self, hidden_size=100, tokens=2, lr=1e-1, var_init=0.01):
        self.hidden_size = hidden_size
        self.lr = lr

        self.weights = dict(
            Wxh=torch.normal(
                0, var_init, (hidden_size, tokens), requires_grad=True
            ),  # input to hidden
            Whh=torch.normal(
                0, var_init, (hidden_size, hidden_size), requires_grad=True
            ),  # hidden to hidden
            Why=torch.normal(
                0, var_init, (tokens, hidden_size), requires_grad=True
            ),  # hidden to output
            bh=torch.normal(
                0, var_init, (hidden_size,), requires_grad=True
            ),  # hidden bias
            by=torch.normal(0, var_init, (tokens,), requires_grad=True),  # output bias
        )

        self.memory = dict(
            mWxh=torch.zeros(
                (hidden_size, tokens), requires_grad=True
            ),  # input to hidden
            mWhh=torch.zeros(
                (hidden_size, hidden_size), requires_grad=True
            ),  # hidden to hidden
            mWhy=torch.zeros(
                (tokens, hidden_size), requires_grad=True
            ),  # hidden to output
            mbh=torch.zeros((hidden_size,), requires_grad=True),  # hidden bias
            mby=torch.zeros((tokens,), requires_grad=True),  # output bias
        )

    def forward(self, inputs):
        Wxh, Whh, Why, bh, by = self.weights.values()
        h = torch.zeros((self.hidden_size,))
        result = []

        for t in range(len(inputs)):
            h = torch.tanh(Wxh @ inputs[t] + Whh @ h + bh)
            p = torch.nn.functional.softmax(Why @ h + by, dim=0)
            result.append((p, h))

        P, H = zip(*result)
        return dict(distribution=torch.stack(P), latent=torch.stack(H))

    def fit(self, data_generator, epochs=-1, callbacks=[]):
        while (epochs := epochs - 1) + 1:
            inputs, targets = data_generator()  # input_length x tokens
            P, H = self.forward(inputs).values()

            # loss = -torch.sum(P * torch.log(torch.sum(P * targets, dim=1))[:, None])
            loss = -torch.sum(
                torch.sum(P, dim=1) * torch.log(torch.sum(P * targets, dim=1))
            )
            loss.backward()

            for callback in callbacks:
                callback(
                    loss=loss, distribution=P, latent=H, inputs=inputs, epoch=epochs
                )

            with torch.no_grad():
                for param, mem in zip(self.weights.values(), self.memory.values()):
                    mem += param.grad.mul(param.grad)
                    param -= self.lr * param.grad / torch.sqrt(mem + 1e-8)
                    param.grad.zero_()

            yield loss.item()


class KerasModel(keras.Model):
    def compute_loss(self, outputs, targets):
        raise NotImplementedError

    def fit(self, data_generator, epochs=-1, lr=1e-3, callbacks=[]):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        while (epochs := epochs - 1) + 1:
            inputs, targets = data_generator()
            outputs = self(inputs)
            loss = self.compute_loss(outputs, targets)

            loss.backward()
            optimizer.step()
            self.zero_grad()

            for callback in callbacks:
                callback(
                    loss=loss,
                    inputs=inputs,
                    targets=targets,
                    logits=outputs,
                    epoch=epochs,
                )


class BinaryLSTMClassifier(KerasModel):
    def __init__(self, input_length, model_scale=1.0):
        size = (numpy.array([256, 128, 64]) * model_scale).round().astype(int)

        inputs = keras.Input(shape=(input_length,))
        x = Embedding(input_dim=2, output_dim=8)(inputs)
        x = LSTM(int(size[0]), return_sequences=False)(x)
        x = Flatten()(x)
        x = Dense(size[1], activation="relu")(x)
        x = Dense(size[2], activation="sigmoid")(x)
        x = Dense(1, activation="sigmoid")(x)
        super().__init__(inputs=inputs, outputs=x)

    def compute_loss(self, outputs, targets):
        return torch.nn.functional.binary_cross_entropy(
            outputs, targets[:, None].float().cuda()
        )


class BinaryConvolutionalClassifier(KerasModel):
    def __init__(self, input_length, model_scale=1):
        size = (numpy.array([16, 32, 64, 128]) * model_scale).round().astype(int)

        inputs = keras.Input(shape=(input_length,))
        x = Embedding(input_dim=2, output_dim=8)(inputs)
        x = Conv1D(filters=size[0], kernel_size=3, activation="relu")(x)
        x = Conv1D(filters=size[1], kernel_size=3, activation="relu")(x)
        x = Conv1D(filters=size[2], kernel_size=3, activation="relu")(x)
        x = Flatten()(x)
        x = Dense(size[3], activation="relu")(x)
        x = Dense(1, activation="sigmoid")(x)
        super().__init__(inputs=inputs, outputs=x)

    def compute_loss(self, outputs, targets):
        return torch.nn.functional.binary_cross_entropy(
            outputs, targets[:, None].float().cuda()
        )


class BinaryAttentionClassifier(KerasModel):
    def __init__(self, input_length, model_scale=1.0):
        size = (numpy.array([256, 128, 64]) * model_scale).round().astype(int)

        inputs = keras.Input(shape=(input_length,))
        x = Embedding(input_dim=2, output_dim=8)(inputs)
        x = Bidirectional(LSTM(int(size[0]), return_sequences=True), merge_mode="sum")(
            x
        )  # Keep sequences for attention
        attention = Attention()([x, x])  # Apply attention to the LSTM outputs
        x = keras.ops.sum(
            attention, axis=1
        )  # Flatten attention output for the dense layers
        x = Dense(size[1], activation="relu")(x)
        x = Dense(size[2], activation="sigmoid")(x)
        x = Dense(1, activation="sigmoid")(x)
        super().__init__(inputs=inputs, outputs=x)

    def compute_loss(self, outputs, targets):
        return torch.nn.functional.binary_cross_entropy(
            outputs, targets[:, None].float().cuda()
        )
