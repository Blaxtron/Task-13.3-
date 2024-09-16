import pickle
import gzip
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

def load_data():
    with gzip.open('data/mnist.pkl.gz', 'rb') as f:
        train_data, val_data, test_data = pickle.load(f, encoding='latin1')
    return train_data, val_data, test_data

def extract_data(dataset):
    try:
        inputs, labels = zip(*dataset)  # Adjust based on actual data structure
    except ValueError as e:
        print("Error extracting data:", e)
        print("Sample data:", dataset[0])
        raise
    return np.array(inputs), np.array(labels)

def prepare_data(data):
    train_data, val_data, test_data = data

    X_train, y_train = extract_data(train_data)
    X_val, y_val = extract_data(val_data)
    X_test, y_test = extract_data(test_data)

    return X_train, y_train, X_val, y_val, X_test, y_test

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x

def main():
    data = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data(data)

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.long))
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=3.0)

    for epoch in range(30):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print progress
        print(f'Epoch {epoch+1} complete')

if __name__ == '__main__':
    main()
