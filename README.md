## Snake Game RL Agent

This repository hosts an implementation of a Reinforcement Learning (RL) agent designed to master the classic Snake game. The project comprises three interconnected Python files:

### 1. game.py
- **Description**: This file defines the game environment using the Pygame library. It encapsulates the entire Snake game logic, including initializing the game state, managing user input, updating the snake's movement, detecting collisions, handling score computation, and rendering the game elements.
- **Key Components**:
  - `SnakeGameAI` Class: Manages the game state and provides methods for game initialization, playing a single step, collision detection, updating the UI, and resetting the game.
  - Constants, Colors, Enums, and Named Tuples: Define game parameters, colors, directions, and point representations.
  - Play Step Method: Controls the main game loop, including user input handling, snake movement, collision detection, score update, and rendering.
- **Usage**: Run this file to start the Snake game environment.

### 2. agent.py
- **Description**: This file implements the RL agent responsible for learning to play the Snake game. It employs a Q-learning algorithm to make decisions based on observed game states, actions, and rewards. The agent interacts with the game environment, learns from experiences, and adjusts its behavior to maximize its score.
- **Key Components**:
  - `Agent` Class: Initializes agent parameters, including exploration rate, discount rate, memory buffer, Q-network model, and trainer. Provides methods for state observation, action selection, memory management, and training.
  - Train Functions: Define the training process, including short-term and long-term memory training steps.
  - Training Loop: Controls the interaction between the agent and the game environment, facilitating learning and decision-making.
- **Usage**: Execute this file to train the RL agent to play the Snake game.

### 3. model.py
- **Description**: This file defines the neural network model used by the RL agent to approximate the Q-values of state-action pairs. It implements a simple feedforward neural network architecture and a trainer class for updating the network parameters during training.
- **Key Components**:
  - `Linear_QNet` Class: Defines a neural network with two linear layers. Implements the forward pass and a method to save the model's state.
  - `QTrainer` Class: Manages the training process, including optimizer initialization, loss computation, and parameter updates.
- **Usage**: Run this file to define the Q-network model and its training process.

### How to Use
1. Run `game.py` to play the Snake game manually.
2. Execute `agent.py` to train the RL agent to play the game autonomously.
3. Use `model.py` to define and train custom neural network models for the agent.

### Dependencies
- Python 3.x
- Pygame
- PyTorch
