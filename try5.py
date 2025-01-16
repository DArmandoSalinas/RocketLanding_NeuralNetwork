import math
import csv
import random


class NeuralNetwork:
    def __init__(self, λ, weights, M, b, neurons): #I initialize the parameters
        self.λ = λ  #lambda
        self.weights = weights  # The list of weights generated
        self.M = M  # Learning rate
        self.b = b  # The bias term thats gonna be applied
        self.neurons = neurons  # Number of neurons 

    # Sigmoid activation function
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-self.λ * x))

    # Defining the derivative of the sigmoid function (yp(1-yp))
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Forward propagation through the network
    def forward_propagate(self, x1, x2):
        v = []  # Pre activation values
        h = []  # Hidden layer outputs
        yp = []  # Output layer outputs

        # Activations for the hidden layer
        for i in range(self.neurons):
            v_hidden = x1 * self.weights[i] + x2 * self.weights[i + self.neurons] + self.b # Multiplying the first an second inputs by their corresponding weights
            v.append(v_hidden) # Storing pre-activation
        for v_hidden in v:
            h.append(self.sigmoid(v_hidden)) # Applying sigmoid activation adn appending

        # Activations for the output layer
        for i in range(2):  # Output layer has 2 neurons
            v_output = sum(h[j] * self.weights[self.neurons * 2 + i * self.neurons + j] for j in range(self.neurons)) + self.b # For each hidden layer neuron, multiplying its activation value by the corresponding weight for the current output neuron plus the bias
            v.append(v_output) # Storing pre activation
            yp_output = self.sigmoid(v_output) # Applying sigmoid activation
            yp.append(yp_output) # Adding to its corresponding list

        return h, yp # Returning the hidden layer and output layer activations

    # To update weights and bias
    def backward_propagate(self, x1, x2, y, h, yp):
        zeta_output = [] # To store the output layer errors
        zeta_hidden = [] # Hidden layer errors

        # Calculating errors for the output layer
        for i in range(2):
            error = y[i] - yp[i]
            delta_output = self.M * error * yp[i] * (1 - yp[i]) # Derivative of sigmoid, i could have added the function but i wanted to explicity write the formula as the professor did in the whiteboard
            zeta_output.append(delta_output) # Appending the errors

        # For the hidden layer is a more complex procedure
        for i in range(self.neurons):
            sum_delta_weights = sum( #Calculates the sum
                zeta_output[j] * self.weights[self.neurons * 2 + j * self.neurons + i] for j in range(2) # Is the propagation from the output layer to the hidden layer according with their specific weights. 
            )
            delta_hidden = self.M * h[i] * (1 - h[i]) * sum_delta_weights # The formula for getting the delta for hidden simplified with the previous procedure of the sum_delta_weights
            zeta_hidden.append(delta_hidden) #Appending to the list

        # Updating the weights for the output layer
        for i in range(2):
            for j in range(self.neurons):
                self.weights[self.neurons * 2 + i * self.neurons + j] += self.M * zeta_output[i] * h[j]

        # Updating weights for the hidden layer 
        for i in range(self.neurons):
            self.weights[i] += self.M * zeta_hidden[i] * x1
            self.weights[i + self.neurons] += self.M * zeta_hidden[i] * x2

        # Updating the bias
        for i in range(2):
            self.b += self.M * zeta_output[i]
        for i in range(self.neurons):
            self.b += self.M * zeta_hidden[i]

    # To compute mean squared error loss
    def compute_loss(self, x1_list, x2_list, y_list):
        total_error = 0 # Initializing the error
        for x1, x2, y in zip(x1_list, x2_list, y_list):
            _, yp = self.forward_propagate(x1, x2) # Predicting output
            for i in range(2):
                total_error += (y[i] - yp[i]) ** 2 # Sum of squared differences
        return total_error / len(x1_list) # Average error

    # Trainning the network
    def train_batch(self, x1_list, x2_list, y_list): 
        for x1, x2, y in zip(x1_list, x2_list, y_list): #Looping through the input and output, using zip(...) to combine the data to tuples
            h, yp = self.forward_propagate(x1, x2) # To perform forward with the inputs
            self.backward_propagate(x1, x2, y, h, yp) # To adjust the weights with new data

    # Splitting the data into to its manage
    def train_test_split(self, file_path, train_ratio=0.7):
        x1_list, x2_list, y_list = [], [], [] # Initializing the lists to store the data
        with open(file_path, newline='') as csvfile:
            reader = csv.reader(csvfile) 
            next(reader) # To skip the header
            for row in reader:
                x1, x2, y1, y2 = map(float, row) # Converting the values to float
                x1_list.append(x1) 
                x2_list.append(x2)
                y_list.append([y1, y2]) #Adding the ouput pair to a list. As there are two outputs, I store it here and work with indexes

        data = list(zip(x1_list, x2_list, y_list)) # zip combines the three lists (x1_list, x2_list, y_list) into tuples (x1, x2, [y1, y2])
        random.shuffle(data)
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size] # data to train
        test_data = data[train_size:] # data to test
        x1_train, x2_train, y_train = zip(*train_data) # Separating the inp and outp and converting to lists
        x1_test, x2_test, y_test = zip(*test_data)
        return list(x1_train), list(x2_train), list(y_train), list(x1_test), list(x2_test), list(y_test)

    # To get into the training and then the testing 
    def train_and_evaluate(self, file_path, epochs):
        x1_train, x2_train, y_train, x1_test, x2_test, y_test = self.train_test_split(file_path)

        for epoch in range(epochs):
            self.train_batch(x1_train, x2_train, y_train)
            train_loss = self.compute_loss(x1_train, x2_train, y_train)
            print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.6f}")
            print(f"Epoch {epoch + 1}: Updated Weights = {self.weights}")
            print(f"Epoch {epoch + 1}: b = {self.b}")

        test_loss = self.compute_loss(x1_test, x2_test, y_test)
        print(f"Final Test Loss: {test_loss:.6f}")


if __name__ == "__main__":
    neurons = 5
    weights = [random.uniform(-1, 1) * (1 / neurons ** 0.5) for _ in range(neurons * 2 + neurons * 2)] #Xavier Init
    b = 1  # Initial bias to be updated 
    # At the beginning I was trying to work with the parameters i got from matlab but my model wasnt working.
    # They were 5 neurons, lambda 0.05 and trainparam of 0.9, but after trying plenty of times, these values showed more movement in the rocket
    nn = NeuralNetwork(λ=0.08, weights=weights, M=1.05, b=b, neurons=neurons) 

    # Path to my CSV file with the normalize data
    csv_file_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\OneDrive_2024-11-25\Assignment Code\normalized_ce889_dataCollection.csv"

    # Trainning and evaluating
    nn.train_and_evaluate(csv_file_path, epochs=150)
