# AlphaZero Player

A game-playing agent that combines a neural network model with Monte Carlo Tree Search (MCTS) to play games.

## Overview

This crate implements a player that uses trained neural network weights along with MCTS to generate self-play games. It's designed to run as a service that can be controlled remotely.

## Features

- **Self-Play Game Generation**: Plays games autonomously using a combination of neural network policy/value predictions and MCTS search
- **Batched GPU Inference**: Efficiently processes multiple neural network evaluations in batches to maximize GPU utilization
- **Dynamic Model Updates**: Hot-swaps neural network weights when new trained models become available
- **Game Data Broadcasting**: Sends completed game data including:
  - All moves played during the game
  - MCTS policy distributions at each move
  - Final game outcome (win/loss/draw)
- **REST API Control**: All interactions handled through a REST API interface

## Operation

The player operates in a start/stop model:
- Sends a **start signal** to begin generating self-play games
- Sends an **end signal** to stop game generation
- Sends **new model weights** to update the neural network used for play

All game data is broadcast via REST endpoints for collection by the training pipeline.