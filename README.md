# AlphaZero

An experimental implementation of [AlphaZero](https://arxiv.org/pdf/1712.01815), a reinforcement learning algorithm that combines Monte Carlo Tree Search (MCTS) with deep neural networks to master games through self-play.

## Background

AlphaZero, developed by DeepMind in 2017, represents a breakthrough in game-playing AI:

- **Neural Network Architecture**: Uses a single deep neural network with two heads:
  - **Policy head**: Outputs move probabilities p(a|s) for each action a given state s
  - **Value head**: Outputs position evaluation v(s) estimating the expected outcome from state s
- **MCTS Enhancement**: The neural network guides MCTS by:
  - Reducing the breadth of search (policy network suggests promising moves)
  - Reducing the depth of search (value network evaluates positions without full rollouts)
- **Training Signal**: Uses only the final game outcome (win/loss/draw) - no intermediate rewards or human annotations

## Overview

This project provides a general-purpose framework for:

- **Training AI agents** to play various games through self-play
- **Playing games** using trained models and MCTS

## Architecture

### Game Interface

To use this framework, games must implement the `AlphaGame` trait, which provides:

- **State**: Represents the current game state
- **Action**: Represents possible moves in the game
- **Controller**: Applies actions to states and manages game rules

### Playing Games

The AI agent combines two components to play:

1. **Learned Model**: A neural network that evaluates positions and suggests moves
2. **MCTS (Monte Carlo Tree Search)**: A search algorithm that explores possible moves

Each game generates training data consisting of:

- **Game outcomes** (win/loss/draw)
- **Move probabilities** (policies learned during play)

### MCTS Algorithm

AlphaZero uses a variant of MCTS with four phases per simulation:

#### 1. Selection

At each node, select the action that maximizes the Upper Confidence Bound (UCB):

$$a_t = \arg\max_a \left( Q(s_t, a) + U(s_t, a) \right)$$

where the exploration term is:

$$U(s, a) = c_{puct} \cdot P(s, a) \cdot \frac{\sqrt{\sum_b N(s, b)}}{1 + N(s, a)}$$

- $Q(s, a)$ = mean action-value (average reward from taking action $a$ in state $s$)
- $U(s, a)$ = exploration bonus favoring actions with high prior probability and low visit count
- $P(s, a)$ = prior probability from the neural network policy head
- $N(s, a)$ = visit count for state-action pair
- $c_{puct}$ = exploration constant controlling exploration vs. exploitation trade-off

#### 2. Expansion

When a leaf node is reached, expand it using the neural network:

$$(P(s, \cdot), v) = f_\theta(s)$$

- $f_\theta$ = neural network with parameters $\theta$
- $P(s, \cdot)$ = policy vector (prior probabilities for all actions)
- $v$ = value estimate for the position

#### 3. Backup

Update statistics for all edges traversed in the simulation:

$$N(s, a) \leftarrow N(s, a) + 1$$

$$W(s, a) \leftarrow W(s, a) + v$$

$$Q(s, a) = \frac{W(s, a)}{N(s, a)}$$

- $W(s, a)$ = total action-value (sum of all values backed up through this edge)
- $v$ = value from neural network evaluation (negated for opponent's perspective)

#### 4. Play

After running $n$ MCTS simulations from the root position, select a move based on visit counts:

$$\pi(a|s_0) = \frac{N(s_0, a)^{1/\tau}}{\sum_b N(s_0, b)^{1/\tau}}$$

- $\pi(a|s_0)$ = move probability for action $a$ from root state $s_0$
- $\tau$ = temperature parameter controlling exploration during play
  - $\tau \to 0$ selects the most visited move (exploitation)
  - $\tau = 1$ samples proportionally to visit counts (exploration)

### Training Process

Training follows a two-phase cycle:

1. **Self-Play Phase**: The current model plays many games against itself, generating training examples
2. **Training Phase**: The neural network is updated using the examples from self-play to improve its position evaluation and move predictions

This cycle repeats iteratively, with each iteration improving the model's strength.

# Components

- Playing <- At least this one is for running in rust.
    - MCTS
    - Receive Model from storage
    - Send games to storage
    - Determine ELO rating.

- Learning
    - Receive games from storage
    - Learn from games
    - Send model to storage
- Storage
- Webview
