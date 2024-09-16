import mnist_loader
import network

# Load MNIST data
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Initialize the network
net = network.Network([784, 30, 10])

# Train the network
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)

# Evaluate the network
accuracy = net.evaluate(test_data)

# Print the accuracy to the console
accuracy_percentage = accuracy / len(test_data) * 100
print(f'Test Accuracy: {accuracy_percentage:.2f}%')

# Write the accuracy to a results.txt file
with open('results.txt', 'w') as file:
    file.write(f'Test Accuracy: {accuracy_percentage:.2f}%\n')