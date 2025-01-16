import csv
from try5 import NeuralNetwork  # Import the NeuralNetwork class

class NeuralNetHolder:
    def __init__(self):
        super().__init__()
        # The path file 
        self.weights_path = r"C:\Users\Salin\OneDrive\Documentos\ESSEX\Neural Networks\trained_weights_try_5.csv"

        # Initializing the neural network with the parameters declared preoviusly BUT b
        # In training the model, i got a final b of 160.32444.... but if put it here my rocket just falls
        self.network = NeuralNetwork(λ=0.08, weights=[], M=1.05, b = 1, neurons=5)

        # Loading the previously trained weights achieved by training the models
        trained_weights = self.load_weights(self.weights_path) #To load the weights
        self.network.weights = trained_weights # Assigning the loaded weights to the neural network

        # Normalization scale 
        self.input_scale = 100

    def load_weights(self, weights_path):

        weights = [] # to store the weights
        with open(weights_path, "r", encoding="utf-8-sig") as file:
            reader = csv.reader(file) 
            for row in reader:
                
                for value in row[0].split(','):  # This splits the rows by commas, to get the individual weights
                    try:
                        # I was struggling at the moment of saving my csv file if the first element was negative because excel thinks i want to do a formula
                        # So, when the first is neg, i add an element like "'" and then i apply the following condition
                        # To remove leading/trailing whitespace and single quotes
                        cleaned_value = value.strip().lstrip("'")
                        weights.append(float(cleaned_value)) 
                    except ValueError:
                        print(f"Skipping invalid value: {value}") # in case i typed an invalid value
        return weights

    def predict(self, input_row):

        x1, x2 = map(float, input_row.split(','))  # Assigning x1,x2 and ensuring its floating type

        # Normalizing the inputs
        x1_normalized = x1 / self.input_scale
        x2_normalized = x2 / self.input_scale

        # After the trainning, for this, only forward_propagate is applied
        _, outputs = self.network.forward_propagate(x1_normalized, x2_normalized)

        # Taking them to their original sacle
        y1 = outputs[0] * self.input_scale
        y2 = outputs[1] * self.input_scale

        # Finally returning the predictions
        return y1, y2
