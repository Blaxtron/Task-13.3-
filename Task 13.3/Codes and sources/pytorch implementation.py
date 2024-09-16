import pickle
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_data():
    # Open and read the compressed MNIST data file
    with gzip.open('Task 13.3/data/mnist.pkl.gz', 'rb') as f:
        train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    return train_data, val_data, test_data

def extract_data(dataset):
    try:
        inputs, labels = dataset  # Since dataset is already a tuple of (inputs, labels)
    except ValueError as e:
        print("Error extracting data:", e)
        print("Sample data:", dataset[:5])  # Inspect the first few elements
        raise
    return np.array(inputs), np.array(labels)

def prepare_data(data):
    # Unpack the training, validation, and test data
    train_data, val_data, test_data = data

    X_train, y_train = extract_data(train_data)
    X_val, y_val = extract_data(val_data)
    X_test, y_test = extract_data(test_data)

    return X_train, y_train, X_val, y_val, X_test, y_test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        # Forward pass through the network
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def accuracy_fn(predictions, labels):
    # Compute the accuracy of the predictions
    _, predicted = torch.max(predictions, 1)
    correct = (predicted == labels).sum().item()
    return correct / len(labels)

def main():
    # Load and prepare the data
    data = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data)

    # Create a DataLoader for the training data
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    # Initialize the network, loss function, and optimizer
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=3.0)

    # Training loop
    for epoch in range(30):
        net.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Calculate accuracy on validation data
        net.eval()
        with torch.no_grad():
            val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                         torch.tensor(y_val, dtype=torch.long))
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False)
            correct = 0
            total = 0
            for inputs, labels in val_loader:
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            val_accuracy = correct / total * 100

        # Print progress and accuracy
        print(f'Epoch {epoch+1} complete - Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
