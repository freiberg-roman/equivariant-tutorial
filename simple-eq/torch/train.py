import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor()])

training_data = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)

test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)


# Define the CNN model
class EqCNN(nn.Module):

    def __init__(self):
        super(EqCNN, self).__init__()
        self.cl1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(1, 3, 3))
        self.max_1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.cl2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(1, 3, 3))
        self.max_2 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        self.cl3 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(1, 5, 5))
        self.dense = nn.Linear(in_features=16, out_features=10)

    def forward(self, x: torch.Tensor):
        x_0 = x
        x_90 = torch.rot90(x, k=1, dims=(2, 3))
        x_180 = torch.rot90(x, k=2, dims=(2, 3))
        x_270 = torch.rot90(x, k=3, dims=(2, 3))

        x = torch.stack([x_0, x_90, x_180, x_270], dim=-3)
        x = nn.functional.silu(self.cl1(x))
        x = self.max_1(x)

        x = nn.functional.silu(self.cl2(x))
        x = self.max_2(x)

        x = nn.functional.silu(self.cl3(x))

        x = x.squeeze()
        x = torch.max(x, dim=-1).values
        logits = self.dense(x)
        return logits


# Training function
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# Test function
def test90(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            X = torch.rot90(X, k=1, dims=(2, 3))
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    model.train()
    print(
        f"Test 90rot Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def test(dataloader, model):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    model.train()
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


# Main function
def main():
    # Create model, loss function and optimizer
    model = EqCNN()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0005)
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False)
    test(test_dataloader, model)
    test90(test_dataloader, model)

    # Train the model
    epochs = 5
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
        test90(test_dataloader, model)
    print("Training complete!")


if __name__ == "__main__":
    main()
