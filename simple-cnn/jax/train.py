import jax
import jax.numpy as jnp
import numpy as np
import optax
import torchvision.datasets as datasets
from flax import linen as nn
from flax.training import train_state
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Load the MNIST dataset
transform = ToTensor()

training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)


# Define the CNN model using Flax
class SimpleCNN(nn.Module):
    def setup(self):
        self.cl1 = nn.Conv(features=8, kernel_size=(3, 3), padding="SAME")
        self.cl2 = nn.Conv(features=16, kernel_size=(3, 3), padding="SAME")
        self.cl3 = nn.Conv(features=16, kernel_size=(7, 7), padding=0)
        self.dense = nn.Dense(features=10)

    def __call__(self, x):
        x = nn.silu(self.cl1(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.silu(self.cl2(x))
        x = nn.max_pool(x, (2, 2), strides=(2, 2))
        x = nn.silu(self.cl3(x))
        x = x.reshape((x.shape[0], -1))  # Flatten
        logits = self.dense(x)
        return logits


# Loss function
def compute_loss(logits, labels):
    return optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=labels
    ).mean()


# Create a TrainState
def create_train_state(learning_rate, model, input_shape):
    params = model.init(jax.random.key(0), jnp.ones(input_shape))["params"]
    tx = optax.adamw(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


# Training step
@jax.jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch[0])
        loss = compute_loss(logits, batch[1])
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (_, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])
    return state, accuracy


# Evaluation function
@jax.jit
def eval_step(params, batch):
    logits = SimpleCNN().apply({"params": params}, batch[0])
    loss = compute_loss(logits, batch[1])
    accuracy = jnp.mean(jnp.argmax(logits, -1) == batch[1])  # type: ignore
    return loss, accuracy


# Train and evaluate the model
def train_and_evaluate(state, train_loader, test_loader, num_epochs):
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc = 0, 0
        for batch in train_loader:
            imgs = jnp.array(batch[0])
            imgs = jnp.transpose(imgs, [0, 2, 3, 1])
            state, acc = train_step(state, (imgs, jnp.array(batch[1])))
            train_acc += acc
        train_acc /= len(train_loader)
        print(f"Epoch {epoch+1}, Train Accuracy: {train_acc:.4f}")

        # Evaluation
        test_loss, test_acc = 0, 0
        for batch in test_loader:
            imgs = jnp.array(batch[0])
            imgs = jnp.transpose(imgs, [0, 2, 3, 1])
            loss, acc = eval_step(state.params, (imgs, jnp.array(batch[1])))
            test_loss += loss
            test_acc += acc
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
        print(
            f"Epoch {epoch+1}, Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}"
        )


def main():
    model = SimpleCNN()

    learning_rate = 0.0005
    num_epochs = 5
    input_shape = (1, 28, 28, 1)  # Batch size of 1 for initializing parameters

    state = create_train_state(learning_rate, model, input_shape)

    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    train_and_evaluate(state, train_loader, test_loader, num_epochs)


if __name__ == "__main__":
    main()
