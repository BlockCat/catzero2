# AlphaZero Player

A high-performance game-playing service that combines neural network evaluation with Monte Carlo Tree Search (MCTS) to generate self-play games for AlphaZero training.

## Overview

This crate implements a distributed self-play service designed to run as part of an AlphaZero training pipeline. It manages parallel game generation using a trained neural network model and MCTS, with efficient GPU utilization through batched inference.

## Architecture

### Components

- **Runner Service**: Manages multiple parallel game-playing threads
  - Configurable number of worker threads
  - Configurable number of parallel games
  - Async task management with cancellation support
  - Lifecycle management (start/stop operations)

- **Batch Service**: Handles neural network inference efficiently
  - Dynamic batching of inference requests
  - Configurable min/max batch sizes
  - Configurable wait time to accumulate batches
  - GPU-accelerated inference with CUDA support

- **REST API**: HTTP interface for service control
  - Status endpoint: Check service state and configuration
  - Start/Stop endpoints: Control game generation
  - Built with Actix Web framework

### Game Play Loop

1. **State Evaluation**: Current game state is sent to the batch service
2. **Inference Request**: State tensor queued for batched inference
3. **Batch Processing**: Multiple requests processed together on GPU
4. **MCTS Search**: Policy and value predictions guide tree search
5. **Move Selection**: Action chosen based on MCTS visit counts
6. **Game Recording**: States, policies, and outcomes saved
7. **Game Broadcast**: Completed games sent to training pipeline

## Features

- **Parallel Self-Play**: Generate multiple games simultaneously across multiple threads
- **Batched GPU Inference**: Maximize GPU utilization by processing inference requests in batches (configurable batch sizes and wait times)
- **Async Architecture**: Built on Tokio for efficient concurrent operations
- **Dynamic Configuration**: Environment-based configuration for all parameters
- **REST API Control**: HTTP endpoints for service management and monitoring
- **Game Data Collection**: Records complete game trajectories with:
  - All game states throughout play
  - MCTS policy distributions at each position
  - Actions taken during the game
  - Final game outcome (winner and value)
- **CUDA Support**: Optional GPU acceleration for neural network inference
- **Multi-Game Support**: Framework designed to support multiple game types (currently Chess)

## Configuration

The service is configured via environment variables:

### Server Configuration
- `HOST`: Server bind address (default: `127.0.0.1`)
- `PORT`: Server port (default: `8080`)
- `WORKERS`: Number of Actix web workers (default: `4`)

### Batch Service Configuration
- `MAX_BATCH_SIZE`: Maximum number of requests per batch (default: `200`)
- `MIN_BATCH_SIZE`: Minimum requests before processing batch (default: `100`)
- `MAX_WAIT_MS`: Maximum milliseconds to wait for batch accumulation (default: `10`)

### Runner Configuration
- `PLAY_CORES`: Number of worker threads for game generation (default: `6`)
- `PARALLEL_GAMES`: Number of games to run in parallel (default: `200`)
- `NUM_ITERATIONS`: MCTS iterations per move (default: `400`)

## REST API Endpoints

### GET `/api/status`
Returns current service status including:
- Running state
- Active game type
- Playing status (whether games are being generated)
- Play info (threads, games in progress, models)
- Batch configuration details

### GET `/api/start_play`
Starts the self-play game generation process.

**Response**: Success message or error if already running

### GET `/api/stop_play`
Stops the self-play game generation process.

**Response**: Success message confirming stop or indication if not running

## Usage

### Building

```bash
# Build with CUDA support (default)
cargo build --release

# Build without CUDA
cargo build --release --no-default-features
```

### Running

```bash
# Run with default configuration
cargo run --release

# Run with custom configuration
HOST=0.0.0.0 PORT=3000 PARALLEL_GAMES=500 cargo run --release
```

### Controlling via API

```bash
# Check status
curl http://localhost:8080/api/status

# Start generating games
curl http://localhost:8080/api/start_play

# Stop generating games
curl http://localhost:8080/api/stop_play
```

## Integration with Training Pipeline

This player service is designed to integrate with a larger AlphaZero training system:

1. **Model Loading**: Loads trained neural network weights on startup
2. **Self-Play Generation**: Produces game data when started via API
3. **Data Output**: Broadcasts completed games (to be implemented: data sink)
4. **Model Updates**: Supports hot-swapping models with new weights (to be implemented)

## Development

### Adding New Games

To add support for a new game:

1. Implement the `AlphaGame` trait from the `mcts` crate
2. Create a game-specific runner implementing the `SingleRunner` trait
3. Add game-specific feature flag in `Cargo.toml`
4. Implement tensor conversion for game states

See `alphazero_chess` for an example implementation.

## Dependencies

- **actix-web**: HTTP server framework
- **tokio**: Async runtime
- **candle-core/candle-nn**: Neural network inference (with optional CUDA)
- **mcts**: Monte Carlo Tree Search implementation
- **alphazero_nn**: AlphaZero neural network architecture
