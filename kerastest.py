import os

# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch
import keras
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
print(device)

# Let's consider a simple MNIST model
def get_model():
    inputs = keras.Input(shape=(784,), name="digits")
    x1 = keras.layers.Dense(64, activation="relu")(inputs)
    x2 = keras.layers.Dense(64, activation="relu")(x1)
    outputs = keras.layers.Dense(10, name="predictions")(x2)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


# Create load up the MNIST dataset and put it in a torch DataLoader
# Prepare the training dataset.
batch_size = 128
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (-1, 784)).astype("float32")
x_test = np.reshape(x_test, (-1, 784)).astype("float32")
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Reserve 10,000 samples for validation.
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Create torch Datasets
train_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_train), torch.from_numpy(y_train)
)
val_dataset = torch.utils.data.TensorDataset(
    torch.from_numpy(x_val), torch.from_numpy(y_val)
)

# Create DataLoaders for the Datasets
train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True
)
val_dataloader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

# Instantiate a torch optimizer
model = get_model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Instantiate a torch loss function
loss_fn = torch.nn.CrossEntropyLoss()


epochs = 3
for epoch in range(epochs):
    for step, (inputs, targets) in enumerate(train_dataloader):
        # Forward pass
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Optimizer variable updates
        optimizer.step()

        # Log every 100 batches.
        if step % 100 == 0:
            print(
                f"Training loss (for 1 batch) at step {step}: {loss.cpu().detach().numpy():.4f}"
            )
            print(f"Seen so far: {(step + 1) * batch_size} samples")
