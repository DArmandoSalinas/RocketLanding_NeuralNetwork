# Rocket Landing with Neural Networks

## Overview
This project demonstrates a custom neural network trained to simulate rocket landings using game data. The network was built from scratch with forward and backward propagation, data normalization, and gradient-based optimization. A **NeuralNetHolder** file is used to streamline testing with saved weights.

## Features
### Custom Neural Network
- Implements forward and backward propagation.
- Uses Xavier initialization for balanced gradients.

### Training and Testing
- Data is split: 70% training, 30% testing.
- Training parameters:
  - Neurons: 5
  - Learning rate (λ): 0.08
  - Train parameter: 1.05.

### Data Normalization
- A normalization file preprocesses game data to ensure inputs are scaled appropriately for efficient training.

### NeuralNetHolder
- Saves final trained weights and biases to a CSV file.
- Loads saved weights for testing and forward propagation without retraining.

## How It Works
1. **Data Normalization**: Scales and formats input data for training.
2. **Forward Propagation**: Computes pre-activation, activation values, and predictions using a sigmoid activation function.
3. **Backward Propagation**:
   - Calculates gradients for the output and hidden layers.
   - Updates weights using gradient descent.
4. **NeuralNetHolder**:
   - Stores trained weights and biases.
   - Allows seamless testing by loading saved parameters.
5. **Loss Function**: Uses Mean Squared Error (MSE) for performance evaluation.

## Training Details
- **Epochs**: 150 (can be reduced as the loss stabilizes).
- **Outputs**: Tracks weights, loss per epoch, and final bias value.

## Results
- The loss decreases significantly during the initial epochs and stabilizes, showing efficient learning.
- The **NeuralNetHolder** enables the rocket to simulate smooth, controlled landings without retraining.

## Demonstration
Watch the rocket's performance:
- [Rocket Landing Video 1](https://youtube.com/shorts/LeMxOrQFNtU?feature=share)
- [Rocket Landing Video 2](https://youtube.com/shorts/9aWIaSHWF5U?feature=share)
- [Rocket Landing Video 3](https://youtube.com/shorts/XIR4RJvBst4?feature=share)

## How to Run
### Steps
1. Clone the repository:
   git clone https://github.com/YOUR_GITHUB_USERNAME/RocketLanding_NeuralNetwork.git
   
2. Download the folder called `OneDrive_2024-11-25`:
   https://drive.google.com/file/d/1k2GwVDpFpAoaO1UfU88ZyHfQCF3xXNdr/view?usp=drive_link

3. To run the game:
   cd venv
   cd Scripts
   Set-ExecutionPolicy Unrestricted -Scope Process
   .\activate
   cd ../../
   python Main.py

4. Collect data using the game interface.

5. Normalize the data

6. Navigate to the project folder:
   cd RocketLanding_NeuralNetwork

7. Train the neural network:
   python try5.py

8. Test with the NeuralNetHolder file:
   python NeuralNetHolder.py

9. Run the game with the trained neural network.

