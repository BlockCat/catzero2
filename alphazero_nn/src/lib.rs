//! AlphaZero Neural Network Implementation
//!
//! This module provides the core neural network architecture for AlphaZero-style game AI.
//! It implements a convolutional neural network with residual blocks, outputting both a
//! policy head (move probabilities) and a value head (position evaluation).
//!
//! # Architecture Overview
//!
//! The network consists of:
//! - **Input Block**: Initial convolutional layer processing game state
//! - **Residual Blocks**: Multiple residual blocks for feature extraction
//! - **Policy Head**: Outputs move probabilities for all legal moves
//! - **Value Head**: Outputs scalar value representing position advantage
//!
//! # Example
//!
//! ```ignore
//! let config = Config::new(119, 19, 256, 3, Shape::from_dims(&[8, 8]), PolicyOutputType::Conv(73));
//! let model = AlphaZeroNN::new(config, vb)?;
//! let (policy, value) = model.forward_t(&input, false)?;
//! ```

use std::collections::HashMap;

use candle_core::{Device, Result, Shape, Tensor};
use candle_nn::{
    BatchNorm, BatchNormConfig, Conv2d, Conv2dConfig, Linear, ModuleT, VarBuilder, batch_norm,
    conv2d, linear, linear_no_bias,
};

/// Trait for game implementations compatible with AlphaZero training and inference.
///
/// Types implementing this trait define how game states are encoded as tensors,
/// and how policy outputs are decoded back to moves.
pub trait AlphaGame {
    /// The type representing a move in this game
    type MoveType;
    /// The type representing a game state
    type GameState: Default + Clone;

    /// Returns the shape of the tensor input for this game
    fn tensor_input_shape() -> Shape;

    /// Returns how policy output should be structured (flat vector or convolutional)
    fn policy_output_type() -> PolicyOutputType;

    /// Returns the number of historical game states to include in the input
    fn history_length() -> usize;

    /// Encodes game states into a batch tensor for neural network input.
    /// The input is a slice of game states, with the last element being the current state.
    /// The amount of states should be smaller or equal to `history_length()`.
    fn encode_game_state(states: &[Self::GameState], device: &Device) -> Tensor;

    /// Decodes policy tensor output into move probabilities
    fn decode_policy_tensor(
        policy_tensor: &candle_core::Tensor,
        legal_moves: &[Self::MoveType],
    ) -> Result<HashMap<Self::MoveType, f32>>;
}

/// Configuration for the AlphaZero neural network architecture.
///
/// Defines the number of layers, channels, and kernel sizes for the network.
pub struct Config {
    /// Number of input channels (e.g., number of features per board square)
    pub n_input_channels: usize,
    /// Number of residual blocks in the network body
    pub n_residual_blocks: usize,
    /// Number of filters/channels throughout the network
    pub n_filters: usize,
    /// Kernel size for convolutional layers (must be odd)
    pub kernel_size: usize,
    /// Spatial dimensions of the game board (e.g., [8, 8] for chess)
    pub game_shape: Shape,
    /// Policy output configuration (Flat vector or Conv output)
    pub policy_output_type: PolicyOutputType,
}

impl Config {
    /// Creates a new network configuration.
    ///
    /// # Arguments
    ///
    /// * `n_input_channels` - Number of input feature channels
    /// * `n_residual_blocks` - Number of residual blocks in the body
    /// * `n_filters` - Number of filters in each layer
    /// * `kernel_size` - Convolution kernel size (should be odd)
    /// * `game_shape` - Board shape as a Shape object
    /// * `policy_output_type` - Policy head output format
    pub fn new(
        n_input_channels: usize,
        n_residual_blocks: usize,
        n_filters: usize,
        kernel_size: usize,
        game_shape: Shape,
        policy_output_type: PolicyOutputType,
    ) -> Self {
        Self {
            n_input_channels,
            n_residual_blocks,
            n_filters,
            kernel_size,
            game_shape,
            policy_output_type,
        }
    }
}

/// The main AlphaZero neural network model.
///
/// Combines an input block, residual body, and dual policy/value heads
/// to implement the AlphaZero architecture.
#[derive(Debug)]
pub struct AlphaZeroNN {
    input_block: InputBlock,
    residual_blocks: Vec<ResidualBlock>,
    policy_head: PolicyHead,
    value_head: ValueHead,
}

impl AlphaZeroNN {
    /// Creates a new AlphaZero neural network with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - Network architecture configuration
    /// * `vb` - Variable builder for initializing model parameters
    ///
    /// # Returns
    ///
    /// A new `AlphaZeroNN` instance with randomly initialized weights
    pub fn new(config: Config, vb: VarBuilder) -> Result<Self> {
        let input_block = InputBlock::new(
            config.n_input_channels,
            config.n_filters,
            config.kernel_size,
            vb.pp("input_block"),
        )?;

        let mut residual_blocks = Vec::with_capacity(config.n_residual_blocks);
        for i in 0..config.n_residual_blocks {
            residual_blocks.push(ResidualBlock::new(
                config.n_filters,
                config.kernel_size,
                vb.pp(format!("residual_block_{}", i)),
            )?);
        }

        let policy_head = PolicyHead::new(
            config.n_filters,
            config.policy_output_type,
            config.kernel_size,
            &config.game_shape,
            vb.pp("policy_head"),
        )?;
        let value_head = ValueHead::new(config.n_filters, &config.game_shape, vb.pp("value_head"))?;

        Ok(Self {
            input_block,
            residual_blocks,
            policy_head,
            value_head,
        })
    }

    /// Forward pass through the network.
    ///
    /// # Arguments
    ///
    /// * `input` - Input tensor of shape [batch_size, n_input_channels, ...game_shape]
    /// * `train` - Whether the network is in training mode (affects batch norm)
    ///
    /// # Returns
    ///
    /// A tuple of (policy_output, value_output) tensors
    pub fn forward_t(&self, input: &Tensor, train: bool) -> Result<(Tensor, Tensor)> {
        let mut x = input.apply_t(&self.input_block, train)?;

        for block in &self.residual_blocks {
            x = block.forward_t(&x, train)?;
        }

        let output = self.policy_head.forward_t(&x, train)?;
        let value = self.value_head.forward_t(&x, train)?;

        Ok((output, value))
    }
}

/// Initial processing block that projects game state to hidden dimension.
///
/// Applies initial convolutions and batch normalization to the input.
#[derive(Debug)]
struct InputBlock {
    input: Conv2d,
    initial_conv: Conv2d,
    initial_batch_norm: BatchNorm,
}

impl InputBlock {
    /// Creates a new input block.
    fn new(
        input_channel: usize,
        output_channel: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        assert!(kernel_size % 2 == 1, "kernel_size must be odd");
        let conv2d_config = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };

        let input = candle_nn::conv::conv2d(
            input_channel,
            output_channel,
            kernel_size,
            conv2d_config,
            vb.pp("input"),
        )?;

        let initial_conv = candle_nn::conv::conv2d(
            output_channel,
            output_channel,
            kernel_size,
            conv2d_config,
            vb.pp("initial_conv"),
        )?;

        let initial_batch_norm = batch_norm(
            output_channel,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("initial_batch_norm"),
        )?;

        Ok(Self {
            input,
            initial_conv,
            initial_batch_norm,
        })
    }
}

impl ModuleT for InputBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        input
            .apply_t(&self.input, train)?
            .apply_t(&self.initial_conv, train)?
            .apply_t(&self.initial_batch_norm, train)?
            .relu()
    }
}
/// A residual block with skip connections.
///
/// Implements the residual connection pattern where input is added to output,
/// allowing for very deep networks.
#[derive(Debug)]
struct ResidualBlock {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    conv2: Conv2d,
    batch_norm2: BatchNorm,
}

impl ResidualBlock {
    /// Creates a new residual block.
    fn new(n_filters: usize, kernel_size: usize, vb: VarBuilder) -> Result<Self> {
        let conv_config = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        let batch_norm_config = BatchNormConfig {
            ..Default::default()
        };
        let conv1 = candle_nn::conv::conv2d(
            n_filters,
            n_filters,
            kernel_size,
            conv_config,
            vb.pp("conv1"),
        )?;

        let batch_norm1 =
            candle_nn::batch_norm::batch_norm(n_filters, batch_norm_config, vb.pp("batch_norm1"))?;

        let conv2 = candle_nn::conv::conv2d(
            n_filters,
            n_filters,
            kernel_size,
            conv_config,
            vb.pp("conv2"),
        )?;
        let batch_norm2 =
            candle_nn::batch_norm::batch_norm(n_filters, batch_norm_config, vb.pp("batch_norm2"))?;

        Ok(Self {
            conv1,
            batch_norm1,
            conv2,
            batch_norm2,
        })
    }
}

impl ModuleT for ResidualBlock {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let x = input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?
            .apply_t(&self.conv2, train)?
            .apply_t(&self.batch_norm2, train)?;
        let x = (input + x)?;
        x.relu()
    }
}

/// Specifies the output format for the policy head.
#[derive(Debug)]
pub enum PolicyOutputType {
    /// Flat vector output with fixed number of moves
    Flat(usize),
    /// Convolutional output with spatial dimensions (one channel per move type)
    Conv(usize),
}

/// Internal enum for policy output layer implementation
#[derive(Debug)]
enum PolicyOutputLayer {
    /// Linear layer for flat output
    Flat(Linear),
    /// Convolutional layer for spatial output
    Conv(Conv2d),
}

/// Policy head that outputs move probabilities.
///
/// Processes features from the residual body and outputs either a flat
/// vector or spatial policy depending on configuration.
#[derive(Debug)]
struct PolicyHead {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    output: PolicyOutputLayer,
}

impl PolicyHead {
    /// Creates a new policy head.
    fn new(
        features: usize,
        output: PolicyOutputType,
        kernel_size: usize,
        shape: &Shape,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv2dConfig {
            padding: kernel_size / 2,
            ..Default::default()
        };
        let conv1 = conv2d(features, features, kernel_size, cfg, vb.pp("conv1"))?;

        let batch_norm1 = batch_norm(
            features,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("batch_norm1"),
        )?;

        let output = match output {
            PolicyOutputType::Flat(size) => {
                let input_dim = features * shape.elem_count();
                PolicyOutputLayer::Flat(linear(input_dim, size, vb.pp("linear"))?)
            }
            PolicyOutputType::Conv(channels) => PolicyOutputLayer::Conv(conv2d(
                features,
                channels,
                kernel_size,
                cfg,
                vb.pp("conv2"),
            )?),
        };

        Ok(Self {
            conv1,
            batch_norm1,
            output,
        })
    }
}

impl ModuleT for PolicyHead {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        let x = input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?;

        match &self.output {
            PolicyOutputLayer::Flat(fc) => x.flatten_from(1)?.apply_t(fc, train),
            PolicyOutputLayer::Conv(conv) => x.apply_t(conv, train),
        }
    }
}

/// Value head that estimates position evaluation.
///
/// Processes features from the residual body through convolution and
/// fully-connected layers to output a single scalar value in [-1, 1].
#[derive(Debug)]
struct ValueHead {
    conv1: Conv2d,
    batch_norm1: BatchNorm,
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
}

impl ValueHead {
    /// Creates a new value head.
    fn new(channels: usize, shape: &Shape, vb: VarBuilder) -> Result<Self> {
        let conv1 = conv2d(
            channels,
            1,
            1,
            Conv2dConfig {
                stride: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;
        let batch_norm1 = batch_norm(
            1,
            BatchNormConfig {
                ..Default::default()
            },
            vb.pp("batch_norm1"),
        )?;

        let fc1 = linear_no_bias(shape.elem_count(), 256, vb.pp("fc1"))?;
        let fc2 = linear_no_bias(256, 1, vb.pp("fc2"))?;

        Ok(Self {
            conv1,
            batch_norm1,
            fc1,
            fc2,
        })
    }
}

impl ModuleT for ValueHead {
    fn forward_t(&self, input: &Tensor, train: bool) -> Result<Tensor> {
        input
            .apply_t(&self.conv1, train)?
            .apply_t(&self.batch_norm1, train)?
            .relu()?
            .flatten_from(1)?
            .apply_t(&self.fc1, train)?
            .relu()?
            .apply_t(&self.fc2, train)?
            .tanh()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use candle_core::{DType, Device, Shape};
    use candle_nn::VarMap;

    #[test]
    fn test_alphazero_nn_forward() -> Result<()> {
        let device = Device::cuda_if_available(0)?;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);

        let config = Config::new(
            119,
            19,
            256,
            3,
            Shape::from_dims(&[8, 8]),
            PolicyOutputType::Conv(73),
        );

        let model = AlphaZeroNN::new(config, vs)?;

        let batch_size = 2;
        // 5 seconds per move.

        let start = Instant::now();

        let input = Tensor::randn(1.0f32, 2.0, &[batch_size, 119, 8, 8], &device)?;

        println!("Input tensor created in {:?}", start.elapsed());

        let (policy_output, value_output) = model.forward_t(&input, false)?;

        println!("Model forward pass completed in {:?}", start.elapsed());

        assert_eq!(
            policy_output.shape(),
            &Shape::from_dims(&[batch_size, 73, 8, 8])
        );
        assert_eq!(value_output.shape(), &Shape::from_dims(&[batch_size, 1]));

        Ok(())
    }

    #[test]
    fn test_residual_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = ResidualBlock::new(64, 3, vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = block.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 64, 8, 8]));

        Ok(())
    }

    #[test]
    fn test_input_block_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let block = InputBlock::new(17, 64, 5, vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 17, 8, 8], &device)?;
        let output = block.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 64, 8, 8]));

        Ok(())
    }

    #[test]
    fn test_policy_head_forward_flat() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = PolicyHead::new(
            64,
            PolicyOutputType::Flat(362),
            3,
            &Shape::from_dims(&[8, 8]),
            vs,
        )?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 362]));

        Ok(())
    }

    #[test]
    fn test_policy_head_forward_conv() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = PolicyHead::new(
            64,
            PolicyOutputType::Conv(73),
            3,
            &Shape::from_dims(&[8, 8]),
            vs,
        )?;
        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 73, 8, 8]));
        Ok(())
    }

    #[test]
    fn test_value_head_forward() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let head = ValueHead::new(64, &Shape::from_dims(&[8, 8]), vs)?;

        let input = Tensor::randn(1.0f32, 2.0, &[1, 64, 8, 8], &device)?;
        let output = head.forward_t(&input, false)?;

        assert_eq!(output.shape(), &Shape::from_dims(&[1, 1]));

        println!("Value head output: {:?}", output.to_vec2::<f32>()?);
        Ok(())
    }
}
