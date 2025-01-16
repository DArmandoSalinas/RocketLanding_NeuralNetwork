# Rocket Landing with Neural Networks

## Overview
This project demonstrates a custom neural network trained to simulate rocket landings using game data. The network was built from scratch with forward and backward propagation, data normalization, and gradient-based optimization. A **NeuralNetHolder** file is used to streamline testing with saved weights.

## Features
- **Custom Neural Network**:
  - Forward and backward propagation.
  - Xavier initialization for balanced gradients.
- **Training and Testing**:
  - Data split: 70% training, 30% testing.
  - Parameters: 5 neurons, learning rate (λ) = 0.08, train param = 1.05.
- **NeuralNetHolder**:
  - Stores and loads pre-trained weights for reuse.
  - Simplifies the testing process without retraining the model.

## How It Works
1. **Data Normalization**: Prepares input data for training.
2. **Forward Propagation**: Computes pre-activation, activation values, and predictions using a sigmoid activation function.
3. **Backward Propagation**:
   - Calculates gradients for the output and hidden layers.
   - Updates weights using gradient descent.
4. **NeuralNetHolder**:
   - Saves the final trained weights and biases to a CSV file.
   - Loads the saved weights for testing and forward propagation.
5. **Loss Function**: Uses Mean Squared Error (MSE) to evaluate performance.

### Training Details
- **Epochs**: 150 (can be reduced as loss stabilizes).
- **Output**: Weights, loss per epoch, and final bias value.

## Results
The loss decreases significantly during the initial epochs and stabilizes, showing efficient learning. Using the **NeuralNetHolder**, the rocket successfully simulates landings with smooth and controlled movements without retraining.

## Demonstration
Watch the rocket's performance:
- [Rocket Landing Video 1](https://youtube.com/shorts/LeMxOrQFNtU?feature=share)
- [Rocket Landing Video 2](https://youtube.com/shorts/9aWIaSHWF5U?feature=share)
- [Rocket Landing Video 3](https://youtube.com/shorts/XIR4RJvBst4?feature=share)

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/RocketLanding_NeuralNetwork.git
2. Dowload the folder called OneDrive_2024-11-25
3. To run the game, run in the terminal the commands that follow:
  
   cd venv

   cd Scripts

   Set-ExecutionPolicy Unrestricted -Scope Process

   .\activate

   cd ../../

   python Main.py

4. Collect data

5. Navigate to the folder:

   cd RocketLanding_NeuralNetwork

6. Train the model by running :

  python try5.py

6. Use NeuralNetHolder to test with saved weights:

   python NeuralNetHolder.py

7. Run the game with the Neural Network trained

